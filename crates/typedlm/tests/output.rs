use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::prelude::*;
use typedlm::{FinishReason, Response, Usage, output_schema, parse_output};

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
enum Urgency {
    Low,
    Medium,
    High,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct LineItem {
    name: String,
    quantity: u8,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
enum Shape {
    Circle { radius: f64 },
    Square(f64),
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Analysis {
    urgency: Urgency,
    note: Option<String>,
    main_item: Option<LineItem>,
    items: Vec<LineItem>,
    shape: Shape,
}

/// Analyse a ticket.
#[derive(TypedLm, Serialize)]
#[lm(output = Analysis, validate = at_least_one_item)]
struct Analyse {
    text: String,
}

fn at_least_one_item(output: &Analysis) -> Result<(), Vec<String>> {
    if output.items.is_empty() {
        return Err(vec!["list at least one item in `items`".into()]);
    }
    Ok(())
}

fn response(content: &str) -> Response {
    response_with(content, FinishReason::Stop)
}

fn response_with(content: &str, finish: FinishReason) -> Response {
    Response {
        content: content.into(),
        model: "test-model".into(),
        finish,
        usage: Usage::default(),
    }
}

fn parse(content: &str) -> Result<Analysis, Error> {
    let _ = Analyse {
        text: String::new(),
    }
    .text;
    parse_output::<Analyse>(&response(content), &output_schema::<Analyse>())
}

const VALID: &str = r#"{
    "urgency": "High",
    "note": null,
    "main_item": {"name": "db", "quantity": 1},
    "items": [{"name": "db", "quantity": 1}],
    "shape": {"Circle": {"radius": 2.0}}
}"#;

#[test]
fn valid_output_parses() {
    let out = parse(VALID).unwrap();
    assert_eq!(out.urgency, Urgency::High);
    assert_eq!(out.shape, Shape::Circle { radius: 2.0 });
}

#[test]
fn fenced_output_with_trailing_comma_parses() {
    let content = format!("Sure!\n```json\n{VALID}\n```").replace("1}]", "1},]");
    assert!(parse(&content).is_ok());
}

#[test]
fn all_violations_are_collected_with_paths() {
    let content = r#"{
        "urgency": "VERY_HIGH",
        "note": 5,
        "main_item": {"name": "db", "quantity": 300},
        "items": [{"name": "db"}],
        "shape": "Triangle"
    }"#;
    let Err(Error::SchemaViolation { violations, .. }) = parse(content) else {
        panic!("expected schema violation");
    };
    let paths: Vec<&str> = violations.iter().map(|v| v.path.as_str()).collect();
    for expected in [
        "$.urgency",
        "$.note",
        "$.main_item.quantity",
        "$.items[0].quantity",
        "$.shape",
    ] {
        assert!(paths.contains(&expected), "missing {expected} in {paths:?}");
    }
    let urgency = violations.iter().find(|v| v.path == "$.urgency").unwrap();
    assert_eq!(
        urgency.message,
        r#"expected one of "Low", "Medium", "High", got "VERY_HIGH""#
    );
}

#[test]
fn missing_required_field_is_reported() {
    let Err(Error::SchemaViolation { violations, .. }) =
        parse(r#"{"urgency": "Low", "items": []}"#)
    else {
        panic!("expected schema violation");
    };
    assert!(
        violations
            .iter()
            .any(|v| v.path == "$.shape" && v.message.contains("missing"))
    );
}

#[test]
fn custom_validation_runs_after_schema() {
    let content = VALID.replace(
        r#""items": [{"name": "db", "quantity": 1}]"#,
        r#""items": []"#,
    );
    let err = parse(&content).unwrap_err();
    assert!(
        matches!(&err, Error::Validation { errors } if errors[0].contains("at least one item"))
    );
    let feedback = err.repair_feedback().unwrap();
    assert!(feedback.contains("- list at least one item"));
}

#[test]
fn repair_feedback_lists_violations() {
    let content = VALID.replace("\"High\"", "\"VERY_HIGH\"");
    let err = parse(&content).unwrap_err();
    assert!(err.is_repairable());
    let feedback = err.repair_feedback().unwrap();
    assert!(feedback.contains(r#"- $.urgency: expected one of "Low", "Medium", "High""#));
}

#[test]
fn refusal_and_truncation_are_not_repairable() {
    let schema = output_schema::<Analyse>();
    let refusal = parse_output::<Analyse>(
        &response_with("I can't help with that.", FinishReason::Refusal),
        &schema,
    )
    .unwrap_err();
    assert!(matches!(refusal, Error::Refusal { .. }));
    assert!(!refusal.is_repairable() && refusal.repair_feedback().is_none());

    let truncated = parse_output::<Analyse>(
        &response_with(r#"{"urgency": "Hi"#, FinishReason::Length),
        &schema,
    )
    .unwrap_err();
    assert!(matches!(truncated, Error::Truncated { .. }));
}

#[test]
fn prose_without_json_is_invalid_json() {
    let err = parse("The ticket seems urgent.").unwrap_err();
    assert!(matches!(err, Error::InvalidJson { .. }));
    assert!(err.repair_feedback().unwrap().contains("not valid JSON"));
}

#[test]
fn display_does_not_leak_model_output() {
    let schema_err = parse(&VALID.replace("\"High\"", "\"SECRET-NAME\"")).unwrap_err();
    assert!(matches!(schema_err, Error::SchemaViolation { .. }));
    assert_eq!(
        schema_err.to_string(),
        "answer violates the output schema at $.urgency"
    );

    // An empty schema accepts anything, so serde reports the unknown variant.
    let deser_err =
        parse_output::<Analyse>(&response(r#""SECRET-NAME""#), &serde_json::json!({})).unwrap_err();
    assert!(matches!(deser_err, Error::Deserialize { .. }));
    assert!(!deser_err.to_string().contains("SECRET-NAME"));
}
