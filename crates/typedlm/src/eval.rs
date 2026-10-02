//! Measuring a program on a labelled dataset (feature `eval`).

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::time::{Duration, Instant};

use futures_util::stream::{self, StreamExt};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::schema::validate;
use crate::{Execution, Failed, Program, Provider, Signature, Usage, output_schema};

/// One input with its expected output.
///
/// `expected` may be partial: only the labelled top-level fields are compared.
#[derive(Debug, Clone)]
pub struct Example<S: Signature> {
    pub input: S,
    pub expected: Map<String, Value>,
}

impl<S: Signature> Example<S>
where
    S::Output: Serialize,
{
    /// An example with a complete expected output.
    pub fn new(input: S, expected: &S::Output) -> Self {
        let Value::Object(expected) =
            serde_json::to_value(expected).expect("output types serialize to JSON")
        else {
            panic!("expected outputs must serialize to a JSON object");
        };
        Self { input, expected }
    }
}

impl<S: Signature> Example<S> {
    /// The expected output as its type, if every field is labelled.
    pub fn expected_output(&self) -> Option<S::Output> {
        serde_json::from_value(Value::Object(self.expected.clone())).ok()
    }
}

#[derive(Debug, Clone)]
pub struct Dataset<S: Signature> {
    pub examples: Vec<Example<S>>,
}

/// A dataset line that does not fit the signature.
#[derive(Debug)]
pub struct DatasetError {
    /// 1-based line number; 0 when the file could not be read.
    pub line: usize,
    pub message: String,
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            0 => write!(f, "{}", self.message),
            line => write!(f, "line {line}: {}", self.message),
        }
    }
}

impl std::error::Error for DatasetError {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Line<S> {
    input: S,
    expected: Map<String, Value>,
}

impl<S: Signature> Dataset<S> {
    /// SHA-256 over the examples (inputs and labels) as canonical JSON, hex-encoded.
    pub fn fingerprint(&self) -> String {
        let examples: Vec<Value> = self
            .examples
            .iter()
            .map(|e| {
                let input = serde_json::to_value(&e.input).expect("inputs serialize to JSON");
                serde_json::json!({ "input": input, "expected": e.expected })
            })
            .collect();
        let canonical = serde_json::to_vec(&examples).expect("JSON values always serialize");
        crate::sha256::hex(&crate::sha256::sha256(&canonical))
    }
}

impl<S: Signature + DeserializeOwned> Dataset<S> {
    pub fn from_jsonl(path: impl AsRef<Path>) -> Result<Self, DatasetError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| DatasetError {
            line: 0,
            message: format!("cannot read {}: {e}", path.display()),
        })?;
        Self::from_jsonl_str(&text)
    }

    /// Parses JSON Lines: one `{"input": …, "expected": …}` object per line; blank
    /// lines are skipped. Each labelled field is checked against the output schema.
    pub fn from_jsonl_str(text: &str) -> Result<Self, DatasetError> {
        let schema = labelled_fields_schema(output_schema::<S>());
        let mut examples = Vec::new();
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let error = |message: String| DatasetError {
                line: index + 1,
                message,
            };
            let parsed: Line<S> = serde_json::from_str(line).map_err(|e| error(e.to_string()))?;
            let violations = validate(&schema, &Value::Object(parsed.expected.clone()));
            if !violations.is_empty() {
                let list: Vec<String> = violations
                    .iter()
                    .map(|v| format!("{}: {}", v.path.replacen('$', "expected", 1), v.message))
                    .collect();
                return Err(error(list.join("; ")));
            }
            examples.push(Example {
                input: parsed.input,
                expected: parsed.expected,
            });
        }
        Ok(Self { examples })
    }
}

/// The output schema with nothing required and unknown fields rejected, so partial
/// labels pass but typos do not.
fn labelled_fields_schema(mut schema: Value) -> Value {
    if let Value::Object(root) = &mut schema {
        root.remove("required");
        root.insert("additionalProperties".into(), Value::Bool(false));
    }
    schema
}

mod regression;
pub use regression::{
    Baseline, BaselineOutcome, Comparison, RegressionError, UPDATE_BASELINES, Verdict, compare,
};

/// Scores one answer against its example, from 0.0 (wrong) to 1.0 (right).
pub trait Metric<S: Signature> {
    fn name(&self) -> &str;
    fn score(&self, example: &Example<S>, actual: &S::Output) -> f64;
}

/// 1.0 if every labelled field equals the answer's, else 0.0.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExactMatch;

/// Share of labelled fields the answer got right.
#[derive(Debug, Clone, Copy, Default)]
pub struct FieldAccuracy;

impl<S: Signature> Metric<S> for ExactMatch
where
    S::Output: Serialize,
{
    fn name(&self) -> &str {
        "exact match"
    }

    fn score(&self, example: &Example<S>, actual: &S::Output) -> f64 {
        let (correct, labelled) = field_hits(&example.expected, &to_object(actual));
        if correct == labelled { 1.0 } else { 0.0 }
    }
}

impl<S: Signature> Metric<S> for FieldAccuracy
where
    S::Output: Serialize,
{
    fn name(&self) -> &str {
        "field accuracy"
    }

    fn score(&self, example: &Example<S>, actual: &S::Output) -> f64 {
        let (correct, labelled) = field_hits(&example.expected, &to_object(actual));
        if labelled == 0 {
            1.0
        } else {
            correct as f64 / labelled as f64
        }
    }
}

fn to_object<T: Serialize>(value: &T) -> Map<String, Value> {
    match serde_json::to_value(value) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    }
}

/// (correct, labelled) over the expected fields.
fn field_hits(expected: &Map<String, Value>, actual: &Map<String, Value>) -> (usize, usize) {
    let correct = expected
        .iter()
        .filter(|(name, want)| actual.get(*name).is_some_and(|got| values_equal(want, got)))
        .count();
    (correct, expected.len())
}

/// JSON equality where `2` and `2.0` are the same number.
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x == y || x.as_f64() == y.as_f64(),
        (Value::Array(xs), Value::Array(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| values_equal(x, y))
        }
        (Value::Object(xs), Value::Object(ys)) => {
            xs.len() == ys.len()
                && xs
                    .iter()
                    .all(|(k, x)| ys.get(k).is_some_and(|y| values_equal(x, y)))
        }
        _ => a == b,
    }
}

/// Result of [`evaluate`]. Holds no model output; failures carry the error's
/// `Display`, which never includes it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub program: String,
    /// SHA-256 of the dataset's examples; reports are only comparable when it matches.
    pub dataset: String,
    /// Model that answered most often.
    pub model: Option<String>,
    pub metric: String,
    pub examples: usize,
    /// Runs per example; scores are the mean over them.
    #[serde(default = "one")]
    pub epochs: usize,
    /// Examples whose runs all scored the same; equals `examples` with one epoch.
    #[serde(default)]
    pub consistent: usize,
    /// Mean metric score; failed runs score 0.
    pub score: f64,
    /// 95 % confidence interval of `score`.
    pub interval: (f64, f64),
    pub fields: Vec<FieldReport>,
    /// Runs valid on the first attempt (out of `examples × epochs`).
    pub valid: usize,
    /// Runs valid after one or more repairs.
    pub repaired: usize,
    /// Runs without a valid answer.
    pub failed: usize,
    /// Example index (0-based) and error, for every failed run.
    pub failures: Vec<(usize, String)>,
    /// Metric score per example (mean over epochs), in dataset order; failed runs score 0.
    pub scores: Vec<f64>,
    pub latency_p50_ms: u64,
    pub latency_p95_ms: u64,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldReport {
    pub name: String,
    pub correct: usize,
    pub labelled: usize,
}

impl FieldReport {
    pub fn accuracy(&self) -> f64 {
        if self.labelled == 0 {
            0.0
        } else {
            self.correct as f64 / self.labelled as f64
        }
    }
}

fn one() -> usize {
    1
}

/// How [`evaluate_with`] runs a dataset.
#[derive(Debug, Clone, Copy)]
pub struct EvalOptions {
    /// Model calls in flight at once (default 4).
    pub concurrency: usize,
    /// Runs per example (default 1). More than one shows how stable the answers are:
    /// the score per example becomes the mean, and the report counts the examples whose
    /// runs agree.
    pub epochs: usize,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            concurrency: 4,
            epochs: 1,
        }
    }
}

/// Example index, elapsed time and outcome of one evaluation run.
type Run<T> = (usize, Duration, Result<Execution<T>, Failed>);

/// Runs every example through `program` once, with at most `concurrency` calls in flight,
/// and scores the answers with `metric`.
pub async fn evaluate<S, P, M>(
    program: &Program<S, P>,
    dataset: &Dataset<S>,
    metric: &M,
    concurrency: usize,
) -> Report
where
    S: Signature + Clone,
    S::Output: Serialize,
    P: Provider,
    M: Metric<S>,
{
    let options = EvalOptions {
        concurrency,
        epochs: 1,
    };
    evaluate_with(program, dataset, metric, options).await
}

/// Like [`evaluate`], with every option, e.g. several epochs per example.
pub async fn evaluate_with<S, P, M>(
    program: &Program<S, P>,
    dataset: &Dataset<S>,
    metric: &M,
    options: EvalOptions,
) -> Report
where
    S: Signature + Clone,
    S::Output: Serialize,
    P: Provider,
    M: Metric<S>,
{
    let epochs = options.epochs.max(1);
    let jobs = (0..epochs).flat_map(|_| dataset.examples.iter().enumerate());
    let mut runs: Vec<Run<S::Output>> = stream::iter(jobs)
        .map(|(index, example)| async move {
            let started = Instant::now();
            let result = program.execute(example.input.clone()).await;
            (index, started.elapsed(), result)
        })
        .buffer_unordered(options.concurrency.max(1))
        .collect()
        .await;
    runs.sort_by_key(|(index, ..)| *index);

    let mut run_scores: Vec<Vec<f64>> = vec![Vec::with_capacity(epochs); dataset.examples.len()];
    let mut latencies = Vec::with_capacity(runs.len());
    let mut fields: BTreeMap<String, FieldReport> = BTreeMap::new();
    let mut models: BTreeMap<String, usize> = BTreeMap::new();
    let (mut valid, mut repaired) = (0, 0);
    let mut failures = Vec::new();
    let mut usage = Usage::default();

    for (index, elapsed, result) in runs {
        let example = &dataset.examples[index];
        latencies.push(elapsed);
        let actual = match result {
            Ok(execution) => {
                usage.input_tokens += execution.usage.input_tokens;
                usage.output_tokens += execution.usage.output_tokens;
                *models.entry(execution.model).or_default() += 1;
                if execution.attempts == 1 {
                    valid += 1
                } else {
                    repaired += 1
                }
                run_scores[index].push(metric.score(example, &execution.output));
                Some(to_object(&execution.output))
            }
            Err(failed) => {
                usage.input_tokens += failed.usage.input_tokens;
                usage.output_tokens += failed.usage.output_tokens;
                run_scores[index].push(0.0);
                failures.push((index, failed.error.to_string()));
                None
            }
        };
        for (name, want) in &example.expected {
            let field = fields.entry(name.clone()).or_insert_with(|| FieldReport {
                name: name.clone(),
                correct: 0,
                labelled: 0,
            });
            field.labelled += 1;
            if actual
                .as_ref()
                .and_then(|a| a.get(name))
                .is_some_and(|got| values_equal(want, got))
            {
                field.correct += 1;
            }
        }
    }

    let consistent = run_scores
        .iter()
        .filter(|s| s.windows(2).all(|w| w[0] == w[1]))
        .count();
    let scores: Vec<f64> = run_scores.iter().map(|s| mean(s)).collect();
    latencies.sort();
    Report {
        program: S::NAME.to_string(),
        dataset: dataset.fingerprint(),
        model: models.into_iter().max_by_key(|(_, n)| *n).map(|(m, _)| m),
        metric: metric.name().to_string(),
        examples: scores.len(),
        epochs,
        consistent,
        score: mean(&scores),
        interval: confidence_interval(&scores),
        fields: fields.into_values().collect(),
        valid,
        repaired,
        failed: failures.len(),
        failures,
        scores,
        latency_p50_ms: percentile(&latencies, 50).as_millis() as u64,
        latency_p95_ms: percentile(&latencies, 95).as_millis() as u64,
        usage,
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// 95 % interval: Wilson score interval for 0/1 scores, normal approximation of
/// the mean otherwise. Both clamped to 0..=1.
fn confidence_interval(scores: &[f64]) -> (f64, f64) {
    const Z: f64 = 1.96;
    let n = scores.len() as f64;
    if scores.is_empty() {
        return (0.0, 0.0);
    }
    let p = mean(scores);
    if scores.iter().all(|&s| s == 0.0 || s == 1.0) {
        let z2 = Z * Z;
        let center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n);
        let half = Z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / (1.0 + z2 / n);
        return ((center - half).max(0.0), (center + half).min(1.0));
    }
    let variance = scores.iter().map(|s| (s - p).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let half = Z * (variance / n).sqrt();
    ((p - half).max(0.0), (p + half).min(1.0))
}

/// Nearest-rank percentile of sorted durations.
fn percentile(sorted: &[Duration], pct: usize) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = (pct * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pct = |x: f64| format!("{:.1} %", x * 100.0);
        writeln!(f, "{}", self.program)?;
        if let Some(model) = &self.model {
            writeln!(f, "model           {model}")?;
        }
        writeln!(f, "examples        {}", self.examples)?;
        if self.epochs > 1 {
            writeln!(
                f,
                "epochs          {}  ({}/{} examples consistent)",
                self.epochs, self.consistent, self.examples
            )?;
        }
        writeln!(
            f,
            "{:<15} {}  (95 % CI {} – {})",
            self.metric,
            pct(self.score),
            pct(self.interval.0),
            pct(self.interval.1)
        )?;
        for field in &self.fields {
            writeln!(
                f,
                "  {:<13} {}  ({}/{})",
                field.name,
                pct(field.accuracy()),
                field.correct,
                field.labelled
            )?;
        }
        let runs = self.examples * self.epochs.max(1);
        let share = |n: usize| {
            pct(if runs == 0 {
                0.0
            } else {
                n as f64 / runs as f64
            })
        };
        writeln!(f, "valid           {}", share(self.valid))?;
        writeln!(f, "repaired        {}", share(self.repaired))?;
        writeln!(f, "failed          {}", share(self.failed))?;
        writeln!(f, "latency p50     {} ms", self.latency_p50_ms)?;
        writeln!(f, "latency p95     {} ms", self.latency_p95_ms)?;
        write!(
            f,
            "tokens          {} in / {} out",
            self.usage.input_tokens, self.usage.output_tokens
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_interval_matches_reference() {
        // Roots of (p - x)^2 = z^2 x (1 - x) / n for 463 of 500, z = 1.96.
        let scores: Vec<f64> = (0..500).map(|i| if i < 463 { 1.0 } else { 0.0 }).collect();
        let (low, high) = confidence_interval(&scores);
        assert!((low - 0.899_665).abs() < 1e-5, "{low}");
        assert!((high - 0.945_839).abs() < 1e-5, "{high}");
    }

    #[test]
    fn all_correct_interval_is_below_one() {
        let (low, high) = confidence_interval(&[1.0; 10]);
        assert!((low - 0.722_460).abs() < 1e-5, "{low}");
        assert_eq!(high, 1.0);
    }

    #[test]
    fn percentiles_use_nearest_rank() {
        let sorted: Vec<Duration> = (1..=20).map(Duration::from_millis).collect();
        assert_eq!(percentile(&sorted, 50), Duration::from_millis(10));
        assert_eq!(percentile(&sorted, 95), Duration::from_millis(19));
        assert_eq!(percentile(&sorted[..1], 95), Duration::from_millis(1));
    }

    #[test]
    fn numbers_compare_by_value() {
        assert!(values_equal(
            &serde_json::json!({"a": [2]}),
            &serde_json::json!({"a": [2.0]})
        ));
        assert!(!values_equal(
            &serde_json::json!(2),
            &serde_json::json!("2")
        ));
    }
}
