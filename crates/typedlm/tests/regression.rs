#![cfg(feature = "eval")]

use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::eval::{
    Baseline, BaselineOutcome, Dataset, ExactMatch, RegressionError, Report, Verdict, compare,
    evaluate,
};
use typedlm::prelude::*;
use typedlm::{
    Capabilities, FinishReason, ProviderError, Request, Response, SchemaDialect, Strategy, Usage,
};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
enum Urgency {
    Low,
    High,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Classification {
    urgency: Urgency,
}

/// Classify a support ticket.
#[derive(TypedLm, Serialize, Deserialize, Clone, Debug)]
#[lm(output = Classification)]
struct Classify {
    text: String,
}

/// "down" → High when `attentive`; otherwise always Low.
struct Model {
    attentive: bool,
}

impl Provider for Model {
    fn capabilities(&self) -> Capabilities {
        Capabilities::new(vec![Strategy::NativeSchema], SchemaDialect::Generic)
    }

    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        let text = request.input["text"].as_str().unwrap();
        let urgency = if self.attentive && text.contains("down") {
            "High"
        } else {
            "Low"
        };
        Ok(Response {
            content: format!(r#"{{"urgency": "{urgency}"}}"#),
            model: "m".into(),
            finish: FinishReason::Stop,
            usage: Usage::default(),
        })
    }
}

/// 20 tickets, 8 of them urgent.
fn dataset() -> Dataset<Classify> {
    let lines: Vec<String> = (0..20)
        .map(|i| {
            let (text, urgency) = if i % 5 < 2 {
                (format!("Service {i} is down"), "High")
            } else {
                (format!("Question {i} about settings"), "Low")
            };
            format!(r#"{{"input": {{"text": "{text}"}}, "expected": {{"urgency": "{urgency}"}}}}"#)
        })
        .collect();
    Dataset::from_jsonl_str(&lines.join("\n")).unwrap()
}

async fn run(attentive: bool, dataset: &Dataset<Classify>) -> Report {
    let program = Program::<Classify, _>::new(Model { attentive });
    evaluate(&program, dataset, &ExactMatch, 4).await
}

fn temp_baseline(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("typedlm-regression-{}", std::process::id()));
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    path
}

#[tokio::test]
async fn first_check_stores_then_compares() {
    let path = temp_baseline("stores.json");
    let baseline = Baseline::at(&path);
    let report = run(true, &dataset()).await;

    assert!(matches!(
        baseline.check(&report).unwrap(),
        BaselineOutcome::Created
    ));
    assert_eq!(baseline.load().unwrap().scores, report.scores);

    let again = run(true, &dataset()).await;
    let BaselineOutcome::Compared(comparison) = baseline.check(&again).unwrap() else {
        panic!("expected a comparison");
    };
    assert_eq!(comparison.verdict, Verdict::NoSignificantChange);
    assert_eq!(comparison.difference, 0.0);
    assert!(comparison.worse.is_empty() && comparison.better.is_empty());
}

#[tokio::test]
async fn significant_drop_is_a_regression_naming_the_examples() {
    let path = temp_baseline("drop.json");
    let baseline = Baseline::at(&path);
    baseline.check(&run(true, &dataset()).await).unwrap();

    let err = baseline.check(&run(false, &dataset()).await).unwrap_err();
    let RegressionError::Regression(comparison) = &err else {
        panic!("expected a regression, got {err}");
    };
    assert_eq!(comparison.verdict, Verdict::Regression);
    assert!((comparison.difference - (-0.4)).abs() < 1e-9);
    assert_eq!(comparison.worse, [0, 1, 5, 6, 10, 11, 15, 16]);
    // Same text as in docs/guides/evaluation.md.
    assert_eq!(
        err.to_string(),
        "exact match 100.0 % → 60.0 % (-40.0 points, 95 % CI -63.5 – -16.5) on 20 examples, \
         8 worse, 0 better: regression; worse examples: 0, 1, 5, 6, 10, 11, 15, 16"
    );
}

#[tokio::test]
async fn improvement_is_reported() {
    let before = run(false, &dataset()).await;
    let after = run(true, &dataset()).await;
    let comparison = compare(&before, &after, 0.0).unwrap();
    assert_eq!(comparison.verdict, Verdict::Improvement);
    assert_eq!(comparison.better.len(), 8);
}

#[tokio::test]
async fn tolerance_absorbs_small_drops() {
    let before = run(true, &dataset()).await;
    let after = run(false, &dataset()).await;
    // Interval of the -40 point drop: -63.5 to -16.5 points.
    assert_eq!(
        compare(&before, &after, 0.1).unwrap().verdict,
        Verdict::Regression
    );
    assert_ne!(
        compare(&before, &after, 0.7).unwrap().verdict,
        Verdict::Regression
    );
}

#[tokio::test]
async fn changed_dataset_is_not_comparable() {
    let before = run(true, &dataset()).await;
    let mut other = dataset();
    other.examples.pop();
    let after = run(true, &other).await;
    let err = compare(&before, &after, 0.0).unwrap_err();
    assert!(
        matches!(err, RegressionError::NotComparable { .. }),
        "{err}"
    );
}

#[tokio::test]
async fn fingerprint_is_stable_and_report_round_trips() {
    let report = run(true, &dataset()).await;
    assert_eq!(report.dataset, dataset().fingerprint());
    assert_eq!(report.dataset.len(), 64);

    let json = serde_json::to_string(&report).unwrap();
    let back: Report = serde_json::from_str(&json).unwrap();
    assert_eq!(back.scores, report.scores);
    assert_eq!(back.interval, report.interval);
}

#[tokio::test]
async fn require_score_uses_the_interval() {
    let report = run(false, &dataset()).await; // 60 %, interval roughly 39 – 78 %
    assert!(
        report.require_score(0.7).is_ok(),
        "70 % lies inside the interval"
    );
    let err = report.require_score(0.9).unwrap_err();
    assert!(matches!(err, RegressionError::BelowMinimum { .. }));
    assert!(
        err.to_string().contains("below the required 90.0 %"),
        "{err}"
    );
}

#[test]
fn unreadable_baseline_is_reported() {
    let path = temp_baseline("broken.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "not json").unwrap();
    let err = Baseline::at(&path).load().unwrap_err();
    assert!(matches!(err, RegressionError::InvalidBaseline { .. }));
}
