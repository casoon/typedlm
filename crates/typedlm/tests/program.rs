use std::collections::VecDeque;
use std::sync::Mutex;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::prelude::*;
use typedlm::{
    Capabilities, FinishReason, ProviderError, Request, Response, SchemaDialect, Strategy, Usage,
};

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
enum Urgency {
    Low,
    High,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Classification {
    urgency: Urgency,
}

/// Classify a support ticket.
#[derive(TypedLm, Serialize)]
#[lm(output = Classification)]
struct Classify {
    text: String,
}

/// Answers from a fixed script and records every request.
struct Scripted {
    strategies: Vec<Strategy>,
    native_schema_visible: bool,
    answers: Mutex<VecDeque<(&'static str, FinishReason)>>,
    requests: Mutex<Vec<Request>>,
}

impl Scripted {
    fn new(answers: &[&'static str]) -> Self {
        Self::with(answers.iter().map(|a| (*a, FinishReason::Stop)).collect())
    }

    fn with(answers: Vec<(&'static str, FinishReason)>) -> Self {
        Self {
            strategies: vec![Strategy::JsonMode, Strategy::NativeSchema],
            native_schema_visible: true,
            answers: Mutex::new(answers.into()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}

impl Provider for &Scripted {
    fn capabilities(&self) -> Capabilities {
        Capabilities::new(self.strategies.clone(), SchemaDialect::OpenAiStrict)
            .native_schema_visible(self.native_schema_visible)
    }

    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        self.requests.lock().unwrap().push(request);
        let (content, finish) = self
            .answers
            .lock()
            .unwrap()
            .pop_front()
            .expect("script exhausted");
        Ok(Response {
            content: content.into(),
            model: "scripted".into(),
            finish,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 2,
            },
        })
    }
}

#[tokio::test]
async fn valid_first_answer() {
    let provider = Scripted::new(&[r#"{"urgency": "High"}"#]);
    let program = Program::<Classify, _>::new(&provider);

    let execution = program
        .execute("Production database is down")
        .await
        .unwrap();
    assert_eq!(
        execution.output,
        Classification {
            urgency: Urgency::High
        }
    );
    assert_eq!(execution.attempts, 1);
    assert_eq!(execution.strategy, Strategy::NativeSchema);
    assert_eq!(execution.model, "scripted");

    let request = &provider.requests()[0];
    assert_eq!(
        request.input,
        serde_json::json!({"text": "Production database is down"})
    );
    assert!(
        request
            .instructions
            .starts_with("Classify a support ticket.")
    );
    assert!(!request.instructions.contains("Output JSON Schema"));
    assert_eq!(
        request.output_schema["additionalProperties"],
        serde_json::json!(false)
    );
    assert!(request.repair.is_empty());
}

#[tokio::test]
async fn invalid_answer_is_repaired_with_feedback() {
    let provider = Scripted::new(&[r#"{"urgency": "VERY_HIGH"}"#, r#"{"urgency": "Low"}"#]);
    let execution = Program::<Classify, _>::new(&provider)
        .execute("x")
        .await
        .unwrap();

    assert_eq!(execution.output.urgency, Urgency::Low);
    assert_eq!(execution.attempts, 2);
    assert_eq!(
        execution.usage,
        Usage {
            input_tokens: 20,
            output_tokens: 4
        }
    );

    let second = &provider.requests()[1];
    assert_eq!(second.repair.len(), 1);
    assert_eq!(second.repair[0].response, r#"{"urgency": "VERY_HIGH"}"#);
    assert!(
        second.repair[0]
            .feedback
            .contains(r#"$.urgency: expected one of "Low", "High""#)
    );
}

#[tokio::test]
async fn gives_up_after_max_repairs() {
    let provider = Scripted::new(&["nope", "still nope"]);
    let err = Program::<Classify, _>::new(&provider)
        .max_repairs(1)
        .run("x")
        .await
        .unwrap_err();
    let Error::RetryLimitExceeded { attempts } = err else {
        panic!("unexpected {err:?}")
    };
    assert_eq!(attempts.len(), 2);
    assert!(
        attempts
            .iter()
            .all(|e| matches!(e, Error::InvalidJson { .. }))
    );
    assert_eq!(provider.requests()[1].repair.len(), 1);
}

#[tokio::test]
async fn without_repairs_the_error_is_returned_directly() {
    let provider = Scripted::new(&["nope"]);
    let err = Program::<Classify, _>::new(&provider)
        .max_repairs(0)
        .run("x")
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidJson { .. }));
}

#[tokio::test]
async fn refusal_is_not_repaired() {
    let provider = Scripted::with(vec![("I won't.", FinishReason::Refusal)]);
    let err = Program::<Classify, _>::new(&provider)
        .run("x")
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Refusal { .. }));
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn prompt_strategies_carry_the_schema_in_the_instructions() {
    let provider = Scripted::new(&[r#"{"urgency": "Low"}"#]);
    Program::<Classify, _>::new(&provider)
        .strategy(Strategy::JsonMode)
        .instructions("Custom.")
        .run("x")
        .await
        .unwrap();
    let request = &provider.requests()[0];
    assert_eq!(request.strategy, Strategy::JsonMode);
    assert!(
        request
            .instructions
            .starts_with("Custom.\n\nOutput JSON Schema:\n")
    );
    assert!(request.instructions.contains("\"urgency\""));
}

#[tokio::test]
async fn failed_execution_reports_cost() {
    let provider = Scripted::new(&["nope", "still nope"]);
    let failed = Program::<Classify, _>::new(&provider)
        .max_repairs(1)
        .execute("x")
        .await
        .unwrap_err();
    assert!(matches!(failed.error, Error::RetryLimitExceeded { .. }));
    assert_eq!(failed.attempts, 2);
    assert_eq!(
        failed.usage,
        Usage {
            input_tokens: 20,
            output_tokens: 4
        }
    );
}

#[tokio::test]
async fn hidden_native_schema_is_also_put_into_the_prompt() {
    let mut provider = Scripted::new(&[r#"{"urgency": "Low"}"#]);
    provider.native_schema_visible = false;
    let execution = Program::<Classify, _>::new(&provider)
        .execute("x")
        .await
        .unwrap();
    assert_eq!(execution.strategy, Strategy::NativeSchema);
    assert!(
        provider.requests()[0]
            .instructions
            .contains("Output JSON Schema:")
    );
}
