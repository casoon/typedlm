use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::prelude::*;
use typedlm::{
    Capabilities, DynProvider, FinishReason, ProviderError, Request, Response, SchemaDialect,
    Strategy, Usage,
};

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
enum Urgency {
    Low,
    Medium,
    High,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct TicketClassification {
    urgency: Urgency,
}

/// Classify a customer support ticket
/// by urgency.
#[derive(TypedLm, Serialize)]
#[lm(output = TicketClassification)]
struct ClassifyTicket {
    text: String,
}

#[derive(TypedLm, Serialize)]
#[lm(output = TicketClassification, name = "classify", description = "Override")]
struct Overridden {
    text: String,
}

#[test]
fn derive_takes_name_and_doc_comment() {
    assert_eq!(ClassifyTicket::NAME, "ClassifyTicket");
    assert_eq!(
        ClassifyTicket::DESCRIPTION,
        "Classify a customer support ticket\nby urgency."
    );
    let _ = ClassifyTicket {
        text: String::new(),
    }
    .text;
}

#[test]
fn derive_attributes_override_defaults() {
    assert_eq!(Overridden::NAME, "classify");
    assert_eq!(Overridden::DESCRIPTION, "Override");
    let _ = Overridden {
        text: String::new(),
    }
    .text;
}

struct Fixed(&'static str);

impl Provider for Fixed {
    fn capabilities(&self) -> Capabilities {
        Capabilities::new(vec![Strategy::NativeSchema], SchemaDialect::Generic)
    }

    async fn complete(&self, _request: Request) -> Result<Response, ProviderError> {
        Ok(Response {
            content: self.0.to_string(),
            model: "fixed".into(),
            finish: FinishReason::Stop,
            usage: Usage::default(),
        })
    }
}

fn request() -> Request {
    Request {
        instructions: ClassifyTicket::DESCRIPTION.into(),
        input: serde_json::to_value(ClassifyTicket {
            text: "db down".into(),
        })
        .unwrap(),
        output_schema: serde_json::to_value(schemars::schema_for!(TicketClassification)).unwrap(),
        strategy: Strategy::NativeSchema,
        options: Default::default(),
        repair: Vec::new(),
    }
}

#[tokio::test]
async fn boxed_providers_are_providers() {
    let providers: Vec<Box<dyn DynProvider>> = vec![
        Box::new(Fixed(r#"{"urgency":"High"}"#)),
        Box::new(Fixed(r#"{"urgency":"Low"}"#)),
    ];

    let mut answers = Vec::new();
    for provider in &providers {
        let response = provider.complete(request()).await.unwrap();
        let out: TicketClassification = serde_json::from_str(&response.content).unwrap();
        answers.push(out.urgency);
    }
    assert_eq!(answers, [Urgency::High, Urgency::Low]);
}
