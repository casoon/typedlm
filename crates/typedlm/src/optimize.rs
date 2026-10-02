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

/// Settings for [`optimize_instructions`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reflective {
    /// Proposals at most (default 6).
    pub iterations: usize,
    /// Model calls the run may spend, roughly: the baseline, each proposal and its
    /// evaluation count, repairs do not (default 400).
    pub max_calls: usize,
    /// Mistakes shown to the teacher per proposal (default 5).
    pub mistakes_shown: usize,
    /// Calls in flight during an evaluation (default 4).
    pub concurrency: usize,
}

impl Default for Reflective {
    fn default() -> Self {
        Self {
            iterations: 6,
            max_calls: 400,
            mistakes_shown: 5,
            concurrency: 4,
        }
    }
}

/// One proposal and how it scored.
#[derive(Debug, Clone, PartialEq)]
pub struct InstructionTrial {
    pub instructions: String,
    /// Validation score; `None` when the teacher gave no usable proposal.
    pub score: Option<f64>,
    pub accepted: bool,
    /// Why the proposal failed, if it did.
    pub error: Option<String>,
}

/// Outcome of [`optimize_instructions`].
#[derive(Debug, Clone)]
pub struct OptimizedInstructions {
    /// The program with the best instructions, with that run as provenance.
    pub compiled: CompiledProgram,
    pub baseline: Report,
    pub best: Report,
    /// Paired comparison of `best` against `baseline`.
    pub comparison: Comparison,
    /// Every proposal, in order.
    pub trials: Vec<InstructionTrial>,
}

/// A mistake shown to the teacher.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Mistake {
    pub input: serde_json::Value,
    pub expected: serde_json::Value,
    /// The program's answer, if it produced one.
    pub actual: Option<serde_json::Value>,
    /// Why it produced none.
    pub error: Option<String>,
}

/// The teacher's task: improve a program's instructions from its mistakes.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct ProposeInstructions {
    /// What the program is for, from its signature.
    pub task: String,
    pub current_instructions: String,
    /// JSON Schema of the program's output.
    pub output_schema: serde_json::Value,
    pub mistakes: Vec<Mistake>,
}

/// The teacher's answer.
#[derive(Debug, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ProposedInstructions {
    /// The complete new instructions.
    pub instructions: String,
}

impl Signature for ProposeInstructions {
    type Output = ProposedInstructions;
    const NAME: &'static str = "ProposeInstructions";
    const DESCRIPTION: &'static str = "You improve the instructions of a language-model \
        program. You get the program's task, its current instructions, the JSON Schema of \
        its output, and examples it got wrong with the expected and the actual output. \
        Write new, complete instructions that keep what works and prevent these mistakes. \
        State general rules; do not quote or refer to the examples. Do not describe the \
        output format beyond what the schema says.";

    fn validate(output: &ProposedInstructions) -> Result<(), Vec<String>> {
        if output.instructions.trim().is_empty() {
            return Err(vec!["instructions must not be empty".into()]);
        }
        Ok(())
    }
}

/// Improves a program's instructions from its mistakes: a teacher model proposes new
/// instructions from the examples the current ones get wrong, each proposal is scored on
/// `validation`, and better ones are kept.
///
/// The program's demonstrations, strategy and generation settings are kept; only its
/// instructions change. The teacher sees validation inputs and the program's answers —
/// pick a teacher those may be sent to.
pub async fn optimize_instructions<S, P, M, T>(
    program: &Program<S, P>,
    teacher: &T,
    validation: &Dataset<S>,
    metric: &M,
    options: Reflective,
) -> Result<OptimizedInstructions, OptimizeError>
where
    S: Signature + Clone,
    S::Output: Serialize,
    P: Provider,
    M: Metric<S>,
    T: Provider,
{
    if validation.examples.is_empty() {
        return Err(OptimizeError::EmptyValidation);
    }
    let per_evaluation = validation.examples.len();
    // The baseline plus at least one proposal and its evaluation.
    if options.max_calls < 2 * per_evaluation + 1 {
        return Err(OptimizeError::BudgetTooSmall {
            max_calls: options.max_calls,
            per_evaluation,
        });
    }

    let eval_options = EvalOptions {
        concurrency: options.concurrency,
        epochs: 1,
    };
    let base = program.compile();
    let schema = crate::output_schema::<S>();
    let proposer = Program::<ProposeInstructions, &T>::new(teacher);

    let (baseline, answers) =
        crate::eval::evaluate_detailed(program, validation, metric, eval_options).await;
    let mut calls = per_evaluation;
    let mut current = (program.effective_instructions(), baseline.clone(), answers);
    let mut trials = Vec::new();

    for iteration in 0..options.iterations {
        if calls + 1 + per_evaluation > options.max_calls {
            break;
        }
        let mistakes = mistakes(
            validation,
            &current.1,
            &current.2,
            options.mistakes_shown,
            iteration,
        );
        if mistakes.is_empty() {
            break;
        }
        calls += 1;
        let request = ProposeInstructions {
            task: S::DESCRIPTION.into(),
            current_instructions: current.0.clone(),
            output_schema: schema.clone(),
            mistakes,
        };
        let proposal = match proposer.run(request).await {
            Ok(proposal) => proposal.instructions,
            Err(error) => {
                trials.push(InstructionTrial {
                    instructions: String::new(),
                    score: None,
                    accepted: false,
                    error: Some(error.to_string()),
                });
                continue;
            }
        };

        let mut candidate = base.clone();
        candidate.instructions = Some(proposal.clone());
        let candidate_program = Program::<S, &P>::from_parts(program.provider(), &candidate);
        let (report, answers) =
            crate::eval::evaluate_detailed(&candidate_program, validation, metric, eval_options)
                .await;
        calls += per_evaluation;
        let accepted = report.score > current.1.score;
        trials.push(InstructionTrial {
            instructions: proposal.clone(),
            score: Some(report.score),
            accepted,
            error: None,
        });
        if accepted {
            current = (proposal, report, answers);
        }
    }

    let (instructions, best, _) = current;
    let mut compiled = base;
    if best.score > baseline.score {
        compiled.instructions = Some(instructions);
    }
    let compiled = compiled.with_provenance(&best);
    let comparison = compare(&baseline, &best, 0.0).expect("same program, metric and dataset");
    Ok(OptimizedInstructions {
        compiled,
        baseline,
        best,
        comparison,
        trials,
    })
}

/// Up to `limit` examples the program got wrong, rotating through them per iteration so
/// the teacher sees different mistakes.
fn mistakes<S: Signature>(
    dataset: &Dataset<S>,
    report: &Report,
    answers: &[(usize, crate::eval::Answer)],
    limit: usize,
    iteration: usize,
) -> Vec<Mistake> {
    let wrong: Vec<&(usize, crate::eval::Answer)> = answers
        .iter()
        .filter(|(index, _)| report.scores.get(*index).is_some_and(|s| *s < 1.0))
        .collect();
    if wrong.is_empty() || limit == 0 {
        return Vec::new();
    }
    let start = (iteration * limit) % wrong.len();
    wrong
        .iter()
        .cycle()
        .skip(start)
        .take(limit.min(wrong.len()))
        .map(|(index, answer)| {
            let example = &dataset.examples[*index];
            Mistake {
                input: serde_json::to_value(&example.input).expect("inputs serialize to JSON"),
                expected: serde_json::Value::Object(example.expected.clone()),
                actual: answer.as_ref().ok().cloned(),
                error: answer.as_ref().err().cloned(),
            }
        })
        .collect()
}
