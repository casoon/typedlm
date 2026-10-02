//! Kept apart from tests/regression.rs: it sets an environment variable, which is
//! shared by all tests running in the same process.
#![cfg(feature = "eval")]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::eval::{Baseline, BaselineOutcome, Dataset, ExactMatch, evaluate};
use typedlm::prelude::*;
use typedlm::{
    Capabilities, FinishReason, ProviderError, Request, Response, SchemaDialect, Strategy, Usage,
};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Answer {
    ok: bool,
}

/// Answer yes or no.
#[derive(TypedLm, Serialize, Deserialize, Clone, Debug)]
#[lm(output = Answer)]
struct Ask {
    text: String,
}

struct Fixed(bool);

impl Provider for Fixed {
    fn capabilities(&self) -> Capabilities {
        Capabilities::new(vec![Strategy::NativeSchema], SchemaDialect::Generic)
    }

    async fn complete(&self, _: Request) -> Result<Response, ProviderError> {
        Ok(Response {
            content: format!(r#"{{"ok": {}}}"#, self.0),
            model: "m".into(),
            finish: FinishReason::Stop,
            usage: Usage::default(),
        })
    }
}

#[tokio::test]
async fn update_variable_replaces_the_baseline_even_after_a_regression() {
    let lines: Vec<String> = (0..10)
        .map(|i| format!(r#"{{"input": {{"text": "q{i}"}}, "expected": {{"ok": true}}}}"#))
        .collect();
    let dataset = Dataset::<Ask>::from_jsonl_str(&lines.join("\n")).unwrap();
    let path = std::env::temp_dir().join(format!("typedlm-update-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let baseline = Baseline::at(&path);

    let good = evaluate(
        &Program::<Ask, _>::new(Fixed(true)),
        &dataset,
        &ExactMatch,
        2,
    )
    .await;
    let bad = evaluate(
        &Program::<Ask, _>::new(Fixed(false)),
        &dataset,
        &ExactMatch,
        2,
    )
    .await;
    baseline.check(&good).unwrap();
    assert!(baseline.check(&bad).is_err());

    // SAFETY: the only test in this binary, so no other thread reads the environment.
    unsafe { std::env::set_var("TYPEDLM_UPDATE_BASELINES", "1") };
    assert!(matches!(
        baseline.check(&bad).unwrap(),
        BaselineOutcome::Updated
    ));
    unsafe { std::env::remove_var("TYPEDLM_UPDATE_BASELINES") };

    assert_eq!(baseline.load().unwrap().score, 0.0);
    assert!(baseline.check(&bad).is_ok());
}
