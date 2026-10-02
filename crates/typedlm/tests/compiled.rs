use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::compiled::FORMAT;
use typedlm::prelude::*;
use typedlm::testing::TestProvider;
use typedlm::{CompileError, CompiledProgram, Strategy};

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
struct Detailed {
    urgency: Urgency,
    reason: String,
}

/// Same name, other output type: what happens when an output type changes.
#[derive(TypedLm, Serialize, Deserialize, Clone, Debug)]
#[lm(output = Detailed, name = "Classify")]
struct ClassifyV2 {
    text: String,
}

fn tuned<P: Provider>(provider: P) -> Program<Classify, P> {
    Program::<Classify, _>::new(provider)
        .instructions("Decide how urgent the ticket is. Outages are High.")
        .demonstration(
            "Checkout is down",
            &Classification {
                urgency: Urgency::High,
            },
        )
        .demonstration(
            "How do I change my avatar?",
            &Classification {
                urgency: Urgency::Low,
            },
        )
        .strategy(Strategy::NativeSchema)
        .temperature(0.0)
        .max_repairs(1)
}

#[tokio::test]
async fn compiled_program_round_trips_and_runs_the_same_requests() {
    let compiled = tuned(TestProvider::valid()).compile();
    assert_eq!(compiled.format, FORMAT);
    assert_eq!(compiled.program, "Classify");
    assert_eq!(compiled.signature.len(), 64);

    let loaded = CompiledProgram::from_json(&compiled.to_json()).unwrap();
    assert_eq!(loaded, compiled);

    let original = TestProvider::answers([r#"{"urgency": "High"}"#]);
    let restored = TestProvider::answers([r#"{"urgency": "High"}"#]);
    tuned(&original).run("db down").await.unwrap();
    loaded
        .program::<Classify, _>(&restored)
        .unwrap()
        .run("db down")
        .await
        .unwrap();

    let (a, b) = (&original.requests()[0], &restored.requests()[0]);
    assert_eq!(a.instructions, b.instructions);
    assert_eq!(a.demonstrations, b.demonstrations);
    assert_eq!(a.options, b.options);
    assert_eq!(b.demonstrations.len(), 2);
    assert_eq!(
        b.demonstrations[0].output,
        serde_json::json!({"urgency": "High"})
    );
    assert!(b.instructions.starts_with("Decide how urgent"));
    assert_eq!(b.options.temperature, Some(0.0));
}

#[test]
fn changed_output_type_is_refused() {
    let compiled = tuned(TestProvider::valid()).compile();
    let err = compiled
        .program::<ClassifyV2, _>(TestProvider::valid())
        .unwrap_err();
    assert!(
        matches!(err, CompileError::SignatureMismatch { .. }),
        "{err}"
    );
    assert!(
        err.to_string().contains("compile the program again"),
        "{err}"
    );
}

#[test]
fn hand_edited_demonstrations_are_checked() {
    let mut compiled = tuned(TestProvider::valid()).compile();
    compiled.demonstrations[1].output = serde_json::json!({"urgency": "Urgent"});
    let err = compiled
        .program::<Classify, _>(TestProvider::valid())
        .unwrap_err();
    assert!(
        matches!(err, CompileError::InvalidDemonstration { index: 1, .. }),
        "{err}"
    );

    let mut compiled = tuned(TestProvider::valid()).compile();
    compiled.demonstrations[0].input = serde_json::json!({"txt": "typo"});
    let err = compiled
        .program::<Classify, _>(TestProvider::valid())
        .unwrap_err();
    assert!(
        matches!(err, CompileError::InvalidDemonstration { index: 0, .. }),
        "{err}"
    );
}

#[test]
fn newer_formats_and_garbage_are_invalid() {
    let mut json: serde_json::Value =
        serde_json::from_str(&tuned(TestProvider::valid()).compile().to_json()).unwrap();
    json["format"] = serde_json::json!(FORMAT + 1);
    let err = CompiledProgram::from_json(&json.to_string()).unwrap_err();
    assert!(err.to_string().contains("newer"), "{err}");
    assert!(matches!(
        CompiledProgram::from_json("{}"),
        Err(CompileError::Invalid { .. })
    ));
}

#[test]
fn save_and_load() {
    let path = std::env::temp_dir().join(format!("typedlm-compiled-{}.json", std::process::id()));
    let compiled = tuned(TestProvider::valid()).compile();
    compiled.save(&path).unwrap();
    assert_eq!(CompiledProgram::load(&path).unwrap(), compiled);
    assert!(std::fs::read_to_string(&path).unwrap().ends_with("}\n"));
}

#[cfg(feature = "eval")]
#[tokio::test]
async fn provenance_comes_from_a_report() {
    use typedlm::eval::{Dataset, ExactMatch, evaluate};

    let dataset = Dataset::<Classify>::from_jsonl_str(
        r#"{"input": {"text": "db down"}, "expected": {"urgency": "High"}}"#,
    )
    .unwrap();
    let provider = TestProvider::answers([r#"{"urgency": "High"}"#]);
    let program = tuned(&provider);
    let mut report = evaluate(&program, &dataset, &ExactMatch, 1).await;
    report.label = Some("qwen3:32b".into());

    let compiled = program.compile().with_provenance(&report);
    let provenance = compiled.provenance.as_ref().unwrap();
    assert_eq!(provenance.model.as_deref(), Some("qwen3:32b"));
    assert_eq!(provenance.score, 1.0);
    assert_eq!(provenance.dataset, dataset.fingerprint());
    assert_eq!(
        CompiledProgram::from_json(&compiled.to_json()).unwrap(),
        compiled
    );
}
