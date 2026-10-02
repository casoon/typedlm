#![cfg(feature = "eval")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::eval::{Dataset, ExactMatch, Example, FieldAccuracy, evaluate};
use typedlm::prelude::*;
use typedlm::{
    Capabilities, FinishReason, ProviderError, Request, Response, SchemaDialect, Strategy, Usage,
};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
enum Urgency {
    Low,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
enum Category {
    Billing,
    Technical,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Classification {
    urgency: Urgency,
    category: Category,
}

/// Classify a support ticket.
#[derive(TypedLm, Serialize, Deserialize, Clone, Debug)]
#[lm(output = Classification)]
struct Classify {
    text: String,
}

/// Keyword "model": "down" → High, "invoice" → Billing; "???" yields no JSON.
/// Tracks how many calls run at once.
#[derive(Default)]
struct Keywords {
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
}

impl Provider for &Keywords {
    fn capabilities(&self) -> Capabilities {
        Capabilities::new(vec![Strategy::NativeSchema], SchemaDialect::Generic)
    }

    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(5)).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);

        let text = request.input["text"].as_str().unwrap();
        let content = if text.contains("???") {
            "I am not sure.".to_string()
        } else {
            let urgency = if text.contains("down") { "High" } else { "Low" };
            let category = if text.contains("invoice") {
                "Billing"
            } else {
                "Technical"
            };
            format!(r#"{{"urgency": "{urgency}", "category": "{category}"}}"#)
        };
        Ok(Response {
            content,
            model: "keywords-1".into(),
            finish: FinishReason::Stop,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        })
    }
}

const DATASET: &str = r#"
{"input": {"text": "Server down"}, "expected": {"urgency": "High", "category": "Technical"}}
{"input": {"text": "Wrong invoice amount"}, "expected": {"urgency": "Low", "category": "Billing"}}

{"input": {"text": "Change my avatar"}, "expected": {"urgency": "Low"}}
{"input": {"text": "Invoice portal down"}, "expected": {"urgency": "High", "category": "Billing"}}
{"input": {"text": "???"}, "expected": {"urgency": "Low"}}
"#;

#[tokio::test]
async fn report_scores_partial_labels_and_failures() {
    let dataset = Dataset::<Classify>::from_jsonl_str(DATASET).unwrap();
    assert_eq!(dataset.examples.len(), 5);
    assert!(dataset.examples[0].expected_output().is_some());
    assert!(
        dataset.examples[2].expected_output().is_none(),
        "partial label"
    );

    let model = Keywords::default();
    let program = Program::<Classify, _>::new(&model).max_repairs(0);
    let report = evaluate(&program, &dataset, &ExactMatch, 2).await;

    // Right: 0, 1, 2 (only urgency labelled). Wrong: 3 (category). Failed: 4.
    assert_eq!(report.examples, 5);
    assert!((report.score - 0.6).abs() < 1e-9);
    assert!(report.interval.0 < 0.6 && report.interval.1 > 0.6);
    assert_eq!((report.valid, report.repaired, report.failed), (4, 0, 1));
    assert_eq!(report.failures[0].0, 4);
    assert_eq!(report.model.as_deref(), Some("keywords-1"));
    assert_eq!(
        report.usage,
        Usage {
            input_tokens: 50,
            output_tokens: 25
        }
    );

    let urgency = report.fields.iter().find(|f| f.name == "urgency").unwrap();
    assert_eq!((urgency.correct, urgency.labelled), (4, 5));
    let category = report.fields.iter().find(|f| f.name == "category").unwrap();
    assert_eq!((category.correct, category.labelled), (2, 3));

    let text = report.to_string();
    assert!(text.contains("exact match     60.0 %"), "{text}");
    assert!(text.contains("  category      66.7 %  (2/3)"), "{text}");
}

#[tokio::test]
async fn concurrency_is_limited() {
    let dataset = Dataset::<Classify>::from_jsonl_str(DATASET).unwrap();
    let model = Keywords::default();
    let program = Program::<Classify, _>::new(&model).max_repairs(0);
    evaluate(&program, &dataset, &FieldAccuracy, 2).await;
    assert_eq!(model.max_in_flight.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn field_accuracy_gives_partial_credit() {
    let dataset = Dataset {
        examples: vec![Example::new(
            Classify {
                text: "Invoice portal down".into(),
            },
            &Classification {
                urgency: Urgency::High,
                category: Category::Billing,
            },
        )],
    };
    let model = Keywords::default();
    let program = Program::<Classify, _>::new(&model);
    let report = evaluate(&program, &dataset, &FieldAccuracy, 4).await;
    assert!((report.score - 0.5).abs() < 1e-9);
}

#[test]
fn dataset_errors_name_the_line_and_field() {
    let typo = r#"{"input": {"text": "a"}, "expected": {"urgency": "High"}}
{"input": {"text": "b"}, "expected": {"urgncy": "High"}}"#;
    let err = Dataset::<Classify>::from_jsonl_str(typo).unwrap_err();
    assert_eq!(err.line, 2);
    assert_eq!(
        err.to_string(),
        "line 2: expected.urgncy: field is not allowed"
    );

    let bad_value = r#"{"input": {"text": "a"}, "expected": {"urgency": "Urgent"}}"#;
    let err = Dataset::<Classify>::from_jsonl_str(bad_value).unwrap_err();
    assert!(
        err.to_string()
            .starts_with("line 1: expected.urgency: expected one of"),
        "{err}"
    );

    let bad_input = r#"{"input": {"txt": "a"}, "expected": {}}"#;
    assert_eq!(
        Dataset::<Classify>::from_jsonl_str(bad_input)
            .unwrap_err()
            .line,
        1
    );

    let missing = Dataset::<Classify>::from_jsonl("does/not/exist.jsonl").unwrap_err();
    assert_eq!(missing.line, 0);
}
