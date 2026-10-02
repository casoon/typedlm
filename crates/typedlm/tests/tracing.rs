//! Checks the trace data `Program::execute` emits, with a minimal subscriber that
//! records span fields and events.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};
use typedlm::prelude::*;
use typedlm::{
    Capabilities, FinishReason, ProviderError, Request, Response, SchemaDialect, Strategy, Usage,
};

type Fields = BTreeMap<String, String>;

#[derive(Default)]
struct Captured {
    spans: Vec<(String, Fields)>,
    events: Vec<Fields>,
}

#[derive(Clone, Default)]
struct Recorder {
    captured: Arc<Mutex<Captured>>,
    next_id: Arc<AtomicU64>,
}

struct Collect<'a>(&'a mut Fields);

impl Visit for Collect<'_> {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().into(), value.into());
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().into(), format!("{value:?}"));
    }
}

impl Subscriber for Recorder {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, attrs: &Attributes<'_>) -> Id {
        let mut fields = Fields::new();
        attrs.record(&mut Collect(&mut fields));
        let mut captured = self.captured.lock().unwrap();
        captured
            .spans
            .push((attrs.metadata().name().into(), fields));
        self.next_id.fetch_add(1, Ordering::SeqCst);
        Id::from_u64(captured.spans.len() as u64)
    }
    fn record(&self, span: &Id, values: &Record<'_>) {
        let mut captured = self.captured.lock().unwrap();
        let fields = &mut captured.spans[span.into_u64() as usize - 1].1;
        values.record(&mut Collect(fields));
    }
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn event(&self, event: &Event<'_>) {
        let mut fields = Fields::new();
        event.record(&mut Collect(&mut fields));
        self.captured.lock().unwrap().events.push(fields);
    }
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}

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
#[derive(TypedLm, Serialize)]
#[lm(output = Classification)]
struct Classify {
    text: String,
}

struct Scripted(Mutex<VecDeque<&'static str>>);

impl Provider for Scripted {
    fn capabilities(&self) -> Capabilities {
        Capabilities::new(vec![Strategy::ToolCall], SchemaDialect::Generic)
    }

    async fn complete(&self, _: Request) -> Result<Response, ProviderError> {
        Ok(Response {
            content: self.0.lock().unwrap().pop_front().unwrap().into(),
            model: "model-2026".into(),
            finish: FinishReason::Stop,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 3,
            },
        })
    }
}

fn scripted(answers: &[&'static str]) -> Scripted {
    Scripted(Mutex::new(answers.iter().copied().collect()))
}

#[tokio::test]
async fn span_carries_genai_fields_and_outcome() {
    let recorder = Recorder::default();
    let _guard = tracing::subscriber::set_default(recorder.clone());

    let program = Program::<Classify, _>::new(scripted(&[
        r#"{"urgency": "SECRET-CUSTOMER"}"#,
        r#"{"urgency": "High"}"#,
    ]));
    program.run("Server down for ACME Corp").await.unwrap();

    let captured = recorder.captured.lock().unwrap();
    let (name, span) = &captured.spans[0];
    assert_eq!(name, "typedlm.execute");
    let expect = [
        ("typedlm.program", "Classify"),
        ("gen_ai.operation.name", "chat"),
        ("typedlm.strategy", "tool_call"),
        ("gen_ai.response.model", "model-2026"),
        ("gen_ai.usage.input_tokens", "20"),
        ("gen_ai.usage.output_tokens", "6"),
        ("typedlm.attempts", "2"),
        ("typedlm.outcome", "repaired"),
    ];
    for (key, value) in expect {
        assert_eq!(span.get(key).map(String::as_str), Some(value), "{key}");
    }
    assert!(!span.contains_key("error.type"));

    let results: Vec<&str> = captured
        .events
        .iter()
        .filter_map(|e| e.get("typedlm.result").map(String::as_str))
        .collect();
    assert_eq!(results, ["schema_violation", "ok"]);

    // Content is off by default: neither input nor answers appear anywhere.
    let everything = format!("{:?}{:?}", captured.spans, captured.events);
    assert!(!everything.contains("ACME"), "{everything}");
    assert!(!everything.contains("SECRET-CUSTOMER"), "{everything}");
}

#[tokio::test]
async fn failure_records_error_type() {
    let recorder = Recorder::default();
    let _guard = tracing::subscriber::set_default(recorder.clone());

    let program = Program::<Classify, _>::new(scripted(&["no json"])).max_repairs(0);
    program.run("x").await.unwrap_err();

    let captured = recorder.captured.lock().unwrap();
    let span = &captured.spans[0].1;
    assert_eq!(span["typedlm.outcome"], "failed");
    assert_eq!(span["error.type"], "invalid_json");
    assert_eq!(span["typedlm.attempts"], "1");
}

#[tokio::test]
async fn content_is_recorded_only_on_request() {
    let recorder = Recorder::default();
    let _guard = tracing::subscriber::set_default(recorder.clone());

    let program =
        Program::<Classify, _>::new(scripted(&[r#"{"urgency": "Low"}"#])).record_content(true);
    program.run("Change avatar").await.unwrap();

    let captured = recorder.captured.lock().unwrap();
    let input = captured
        .events
        .iter()
        .find_map(|e| e.get("typedlm.input"))
        .unwrap();
    assert!(input.contains("Change avatar"));
    let response = captured
        .events
        .iter()
        .find_map(|e| e.get("typedlm.response"))
        .unwrap();
    assert_eq!(response, r#"{"urgency": "Low"}"#);
}
