//! Choosing worked examples automatically (feature `optimize`).
//!
//! Given a pool of candidate demonstrations and a validation dataset, the optimizer searches
//! for the set of `k` demonstrations that scores best, and returns it as a
//! [`CompiledProgram`]. The search is pathwise's budgeted local search: start from a seeded
//! choice, swap one demonstration at a time, keep a swap when the validation score rises,
//! stop at a local optimum or when the budget is spent.
//!
//! ```ignore
//! let pool = demonstrations_from_labels(&train);
//! let result = optimize_few_shot(&program, &pool, &validation, &ExactMatch, FewShot::default()).await?;
//! println!("{}", result.comparison);
//! result.compiled.save("classify.typedlm.json")?;
//! ```
//!
//! The winner is selected on the validation set, so its score there is optimistic. Confirm it
//! on a separate test set before relying on the number.

use std::fmt;

use pathwise::Problem;
use pathwise::optimization::{
    AsyncOptimizationProblem, BudgetedSearchOptions, budgeted_local_search,
};
use serde::Serialize;

use crate::compiled::CompiledProgram;
use crate::eval::{Comparison, Dataset, EvalOptions, Metric, Report, compare, evaluate_with};
use crate::{Demonstration, Program, Provider, Signature};

/// Settings for [`optimize_few_shot`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FewShot {
    /// Demonstrations in the resulting program (default 4).
    pub k: usize,
    /// Model calls the search may spend, roughly: each evaluation costs one call per
    /// validation example, plus repairs (default 400).
    pub max_calls: usize,
    /// Seed for the starting choice and the order of swaps (default 0).
    pub seed: u64,
    /// Calls in flight during an evaluation (default 4).
    pub concurrency: usize,
}

impl Default for FewShot {
    fn default() -> Self {
        Self {
            k: 4,
            max_calls: 400,
            seed: 0,
            concurrency: 4,
        }
    }
}

/// One evaluated set of demonstrations.
#[derive(Debug, Clone, PartialEq)]
pub struct Trial {
    /// Indices into the pool, sorted.
    pub demonstrations: Vec<usize>,
    pub score: f64,
    /// Whether the search moved to this set.
    pub accepted: bool,
}

/// Outcome of [`optimize_few_shot`].
#[derive(Debug, Clone)]
pub struct Optimized {
    /// The program with the best demonstrations, with that run as provenance.
    pub compiled: CompiledProgram,
    /// The program as given, on the validation set.
    pub baseline: Report,
    /// The best demonstrations, on the validation set.
    pub best: Report,
    /// Paired comparison of `best` against `baseline`.
    pub comparison: Comparison,
    /// Every evaluated set, in order; the first is the starting choice.
    pub trials: Vec<Trial>,
    /// `false` when the budget ran out before no swap helped any more.
    pub local_optimum: bool,
}

/// Why the optimizer could not run.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OptimizeError {
    /// Fewer candidates than demonstrations to choose.
    PoolTooSmall {
        pool: usize,
        k: usize,
    },
    EmptyValidation,
    /// The budget does not cover a single evaluation besides the baseline.
    BudgetTooSmall {
        max_calls: usize,
        per_evaluation: usize,
    },
}

impl fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PoolTooSmall { pool, k } => {
                write!(f, "{pool} candidate demonstrations, need at least {k}")
            }
            Self::EmptyValidation => write!(f, "the validation dataset is empty"),
            Self::BudgetTooSmall {
                max_calls,
                per_evaluation,
            } => write!(
                f,
                "a budget of {max_calls} calls does not cover one evaluation \
                 ({per_evaluation} calls)"
            ),
        }
    }
}

impl std::error::Error for OptimizeError {}

/// Candidates from labelled examples whose expected output is complete.
pub fn demonstrations_from_labels<S>(dataset: &Dataset<S>) -> Vec<Demonstration>
where
    S: Signature,
{
    dataset
        .examples
        .iter()
        .filter(|e| e.expected_output().is_some())
        .map(|e| Demonstration {
            input: serde_json::to_value(&e.input).expect("inputs serialize to JSON"),
            output: serde_json::Value::Object(e.expected.clone()),
        })
        .collect()
}

/// Candidates from a program's own answers: runs `teacher` on every example and keeps the
/// answers `metric` scores 1. Works with partial labels, and with a stronger model as
/// teacher for a smaller one.
pub async fn bootstrap_demonstrations<S, P, M>(
    teacher: &Program<S, P>,
    dataset: &Dataset<S>,
    metric: &M,
) -> Vec<Demonstration>
where
    S: Signature + Clone,
    S::Output: Serialize,
    P: Provider,
    M: Metric<S>,
{
    let mut pool = Vec::new();
    for example in &dataset.examples {
        let Ok(output) = teacher.run(example.input.clone()).await else {
            continue;
        };
        if metric.score(example, &output) >= 1.0 {
            pool.push(Demonstration {
                input: serde_json::to_value(&example.input).expect("inputs serialize to JSON"),
                output: serde_json::to_value(&output).expect("outputs serialize to JSON"),
            });
        }
    }
    pool
}

/// Searches `pool` for the `options.k` demonstrations that score best on `validation`.
///
/// The program's other settings — instructions, strategy, generation options — are kept;
/// its demonstrations are replaced. The baseline is the program as given.
pub async fn optimize_few_shot<S, P, M>(
    program: &Program<S, P>,
    pool: &[Demonstration],
    validation: &Dataset<S>,
    metric: &M,
    options: FewShot,
) -> Result<Optimized, OptimizeError>
where
    S: Signature + Clone,
    S::Output: Serialize,
    P: Provider,
    M: Metric<S>,
{
    if pool.len() < options.k {
        return Err(OptimizeError::PoolTooSmall {
            pool: pool.len(),
            k: options.k,
        });
    }
    if validation.examples.is_empty() {
        return Err(OptimizeError::EmptyValidation);
    }
    let per_evaluation = validation.examples.len();
    let max_evaluations = options.max_calls / per_evaluation;
    if max_evaluations == 0 {
        return Err(OptimizeError::BudgetTooSmall {
            max_calls: options.max_calls,
            per_evaluation,
        });
    }

    let eval_options = EvalOptions {
        concurrency: options.concurrency,
        epochs: 1,
    };
    let baseline = evaluate_with(program, validation, metric, eval_options).await;

    let search = DemonstrationSearch {
        provider: program.provider(),
        base: program.compile(),
        pool,
        validation,
        metric,
        k: options.k,
        initial: seeded_choice(pool.len(), options.k, options.seed),
        eval_options,
    };
    let solution = budgeted_local_search(
        &search,
        BudgetedSearchOptions {
            max_evaluations,
            seed: options.seed,
        },
    )
    .await;

    let best = solution.score;
    let compiled = search
        .with_demonstrations(&solution.state)
        .with_provenance(&best);
    let comparison = compare(&baseline, &best, 0.0).expect("same program, metric and dataset");
    let trials = solution
        .evaluations
        .into_iter()
        .map(|e| Trial {
            demonstrations: e.state,
            score: e.score.score,
            accepted: e.accepted,
        })
        .collect();
    Ok(Optimized {
        compiled,
        baseline,
        best,
        comparison,
        trials,
        local_optimum: solution.local_optimum,
    })
}

/// `k` distinct indices below `n`, sorted, chosen by `seed`.
fn seeded_choice(n: usize, k: usize, seed: u64) -> Vec<usize> {
    // SplitMix64: a well-mixed stream from any seed, for a Fisher–Yates prefix.
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (next() % (n - i) as u64) as usize;
        indices.swap(i, j);
    }
    let mut chosen = indices[..k].to_vec();
    chosen.sort_unstable();
    chosen
}

/// The search space: sets of `k` pool indices; a move swaps one chosen index for an
/// unchosen one.
struct DemonstrationSearch<'a, S: Signature, P, M> {
    provider: &'a P,
    base: CompiledProgram,
    pool: &'a [Demonstration],
    validation: &'a Dataset<S>,
    metric: &'a M,
    k: usize,
    initial: Vec<usize>,
    eval_options: EvalOptions,
}

impl<S: Signature, P, M> DemonstrationSearch<'_, S, P, M> {
    fn with_demonstrations(&self, chosen: &[usize]) -> CompiledProgram {
        let mut compiled = self.base.clone();
        compiled.demonstrations = chosen.iter().map(|&i| self.pool[i].clone()).collect();
        compiled
    }
}

impl<S: Signature, P, M> Problem for DemonstrationSearch<'_, S, P, M> {
    type State = Vec<usize>;
    /// (position in the chosen set, pool index to put there)
    type Move = (usize, usize);

    fn initial(&self) -> Vec<usize> {
        self.initial.clone()
    }

    fn moves(&self, state: &Vec<usize>) -> impl Iterator<Item = (usize, usize)> {
        let unchosen: Vec<usize> = (0..self.pool.len())
            .filter(|i| !state.contains(i))
            .collect();
        (0..self.k).flat_map(move |slot| unchosen.clone().into_iter().map(move |item| (slot, item)))
    }

    fn apply(&self, state: &Vec<usize>, &(slot, item): &(usize, usize)) -> Vec<usize> {
        let mut next = state.clone();
        next[slot] = item;
        next.sort_unstable();
        next
    }

    fn is_goal(&self, _: &Vec<usize>) -> bool {
        false
    }
}

impl<S, P, M> AsyncOptimizationProblem for DemonstrationSearch<'_, S, P, M>
where
    S: Signature + Clone,
    S::Output: Serialize,
    P: Provider,
    M: Metric<S>,
{
    type Score = Report;

    async fn evaluate(&self, state: &Vec<usize>) -> Report {
        let compiled = self.with_demonstrations(state);
        let program = Program::<S, &P>::from_parts(self.provider, &compiled);
        evaluate_with(&program, self.validation, self.metric, self.eval_options).await
    }

    fn is_improvement(&self, candidate: &Report, incumbent: &Report) -> bool {
        candidate.score > incumbent.score
    }
}

#[cfg(test)]
mod tests {
    use super::seeded_choice;

    #[test]
    fn seeded_choices_are_distinct_sorted_and_reproducible() {
        let a = seeded_choice(10, 4, 7);
        assert_eq!(a.len(), 4);
        assert!(a.windows(2).all(|w| w[0] < w[1]));
        assert!(a.iter().all(|&i| i < 10));
        assert_eq!(a, seeded_choice(10, 4, 7));
        assert_ne!(seeded_choice(10, 4, 7), seeded_choice(10, 4, 8));
        assert_eq!(seeded_choice(3, 3, 1), vec![0, 1, 2]);
    }
}
