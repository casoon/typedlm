use std::sync::atomic::{AtomicUsize, Ordering};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::prelude::*;
use typedlm::testing::{FnProvider, Recorder, Replay, TestProvider};
use typedlm::{ProviderError, Strategy};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct LineItem {
    description: String,
    #[schemars(range(min = 1))]
    quantity: u32,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Invoice {
    number: String,
    items: Vec<LineItem>,
    total_cents: u64,
    note: Option<String>,
    urgency: Urgency,
    previous: Option<Box<Invoice>>,
}

/// Extract an invoice.
#[derive(TypedLm, Serialize)]
#[lm(output = Invoice)]
struct Extract {
    text: String,
}

#[tokio::test]
async fn fixed_answers_and_recorded_requests() {
    let provider = TestProvider::answers([r#"{"urgency": "High"}"#]);
    let out = Program::<Classify, _>::new(&provider)
        .run("db down")
        .await
        .unwrap();
    assert_eq!(out.urgency, Urgency::High);
    assert_eq!(provider.requests()[0].input["text"], "db down");

    let err = Program::<Classify, _>::new(&provider)
        .run("again")
        .await
        .unwrap_err();
    assert!(err.to_string().contains("no answers left"), "{err}");
}

#[tokio::test]
async fn valid_answers_fit_any_signature() {
    let invoice = Program::<Extract, _>::new(TestProvider::valid())
        .execute("x")
        .await
        .unwrap();
    assert_eq!(invoice.attempts, 1);
    assert_eq!(invoice.output.urgency, Urgency::Low);
    assert!(invoice.output.items.is_empty() && invoice.output.note.is_none());

    // Strict dialect too: every property required, optional ones nullable.
    let strict = TestProvider::valid().capabilities(typedlm::Capabilities::new(
        vec![Strategy::NativeSchema],
        typedlm::SchemaDialect::OpenAiStrict,
    ));
    assert!(Program::<Extract, _>::new(strict).run("x").await.is_ok());
}

#[tokio::test]
async fn closures_answer_by_input() {
    let provider = FnProvider::new(|request| {
        let urgent = request.input["text"].as_str().unwrap().contains("down");
        Ok(format!(
            r#"{{"urgency": "{}"}}"#,
            if urgent { "High" } else { "Low" }
        ))
    });
    let program = Program::<Classify, _>::new(provider);
    assert_eq!(
        program.run("server down").await.unwrap().urgency,
        Urgency::High
    );
    assert_eq!(program.run("question").await.unwrap().urgency, Urgency::Low);
}

#[tokio::test]
async fn recorded_interactions_replay_offline() {
    let path = std::env::temp_dir().join(format!("typedlm-cassette-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let calls = AtomicUsize::new(0);
    let live = FnProvider::new(|_| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(r#"{"urgency": "High"}"#.into())
    });

    let recorder = Recorder::new(&path, live).unwrap();
    let recorded = Program::<Classify, _>::new(&recorder);
    recorded.run("db down").await.unwrap();
    recorded.run("printer jam").await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    let replay = Replay::open(&path).unwrap();
    assert_eq!(replay.len(), 2);
    let program = Program::<Classify, _>::new(&replay);
    assert_eq!(program.run("db down").await.unwrap().urgency, Urgency::High);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "replay never calls the live provider"
    );

    let err = program.run("never recorded").await.unwrap_err();
    assert!(
        matches!(err, Error::Provider(ProviderError::Recording(ref m)) if m.contains("no recording")),
        "{err}"
    );
    // Changed instructions change the request, so the recording no longer matches.
    let changed = Program::<Classify, _>::new(&replay).instructions("Other words.");
    assert!(changed.run("db down").await.is_err());
}

#[tokio::test]
async fn recording_keeps_earlier_interactions() {
    let path =
        std::env::temp_dir().join(format!("typedlm-cassette-keep-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let live = || FnProvider::new(|_| Ok(r#"{"urgency": "Low"}"#.into()));

    let first = Recorder::new(&path, live()).unwrap();
    Program::<Classify, _>::new(&first).run("a").await.unwrap();
    let second = Recorder::new(&path, live()).unwrap();
    Program::<Classify, _>::new(&second).run("b").await.unwrap();

    assert_eq!(Replay::open(&path).unwrap().len(), 2);
}
