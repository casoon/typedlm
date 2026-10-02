use std::sync::Mutex;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::prelude::*;
use typedlm::testing::FnProvider;
use typedlm::tools::{Action, Effect, Policy, ToolError, ToolProgram, tool_specs};
use typedlm::{Capabilities, SchemaDialect, Strategy, output_schema};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct OrderId(u64);

/// What the assistant does next.
#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
enum SupportAction {
    /// Look up an order.
    FindOrder {
        order_id: OrderId,
    },
    /// Cancel an order. Changes data.
    CancelOrder {
        order_id: u64,
        reason: String,
    },
    /// Hand over to a human.
    Escalate,
    Wait,
}

/// Decide what to do with a customer message.
#[derive(TypedLm, Serialize)]
#[lm(output = SupportAction)]
struct Decide {
    message: String,
}

#[derive(Default)]
struct Shop {
    cancelled: Mutex<Vec<u64>>,
    user: &'static str,
}

impl Action for SupportAction {
    type Context = Shop;
    type Output = String;
    type Error = String;

    fn effect(&self) -> Effect {
        match self {
            Self::CancelOrder { .. } => Effect::Write,
            _ => Effect::Read,
        }
    }

    async fn execute(self, shop: &Shop) -> Result<String, String> {
        match self {
            Self::FindOrder {
                order_id: OrderId(0),
            } => Err("no order 0".into()),
            Self::FindOrder { order_id } => Ok(format!("order {} is in transit", order_id.0)),
            Self::CancelOrder { order_id, .. } => {
                shop.cancelled.lock().unwrap().push(order_id);
                Ok(format!("order {order_id} cancelled"))
            }
            Self::Escalate => Ok("handed over".into()),
            Self::Wait => Ok("waiting".into()),
        }
    }
}

/// Refuses everything for "guest"; confirms writes when `confirm` is set.
struct ShopPolicy {
    confirm: bool,
}

impl Policy<SupportAction> for ShopPolicy {
    fn authorize(&self, _: &SupportAction, shop: &Shop) -> Result<(), String> {
        if shop.user == "guest" {
            Err("guests cannot act on orders".into())
        } else {
            Ok(())
        }
    }

    async fn confirm(&self, _: &SupportAction, _: &Shop) -> bool {
        self.confirm
    }
}

/// A model that always picks `answer`, offering native tool calls.
fn model(
    answer: &'static str,
) -> FnProvider<impl Fn(&typedlm::Request) -> Result<String, typedlm::ProviderError> + use<>> {
    FnProvider::new(move |_| Ok(answer.into())).capabilities(Capabilities::new(
        vec![Strategy::NativeSchema, Strategy::ToolCall],
        SchemaDialect::Generic,
    ))
}

fn shop(user: &'static str) -> Shop {
    Shop {
        user,
        ..Shop::default()
    }
}

#[test]
fn every_variant_becomes_a_tool() {
    let specs = tool_specs(&output_schema::<Decide>()).unwrap();
    let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["Wait", "FindOrder", "CancelOrder", "Escalate"]);

    let find = &specs[1];
    assert_eq!(find.description.as_deref(), Some("Look up an order."));
    assert!(!find.unit);
    assert_eq!(find.parameters["required"], serde_json::json!(["order_id"]));
    assert!(
        find.parameters["$defs"]["OrderId"].is_object(),
        "references stay resolvable"
    );

    let escalate = &specs[3];
    assert!(escalate.unit);
    assert_eq!(escalate.parameters["type"], "object");
    assert_eq!(
        escalate.output(serde_json::json!({})),
        serde_json::json!("Escalate")
    );
    assert_eq!(
        find.output(serde_json::json!({"order_id": 7})),
        serde_json::json!({"FindOrder": {"order_id": 7}})
    );
}

#[test]
fn structs_and_tuple_variants_are_not_split() {
    #[derive(JsonSchema)]
    #[allow(dead_code)]
    enum Shape {
        Circle { r: f64 },
        Square(f64),
    }
    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct Plain {
        a: u8,
    }
    assert!(tool_specs(&schemars::schema_for!(Shape).to_value()).is_none());
    assert!(tool_specs(&schemars::schema_for!(Plain).to_value()).is_none());
}

#[tokio::test]
async fn read_actions_run_after_authorization() {
    let assistant = ToolProgram::new(
        Program::<Decide, _>::new(model(r#"{"FindOrder": {"order_id": 4711}}"#)),
        ShopPolicy { confirm: false },
    );
    let outcome = assistant
        .run("Where is my order 4711?", &shop("alice"))
        .await
        .unwrap();
    assert_eq!(outcome.action, "FindOrder");
    assert_eq!(outcome.effect, Effect::Read);
    assert_eq!(outcome.output, "order 4711 is in transit");

    let unit = ToolProgram::new(
        Program::<Decide, _>::new(model(r#""Escalate""#)),
        ShopPolicy { confirm: false },
    );
    assert_eq!(
        unit.run("I want a human", &shop("alice"))
            .await
            .unwrap()
            .action,
        "Escalate"
    );
}

#[tokio::test]
async fn writes_need_confirmation() {
    let cancel = r#"{"CancelOrder": {"order_id": 4711, "reason": "changed my mind"}}"#;
    let context = shop("alice");

    let unconfirmed = ToolProgram::new(
        Program::<Decide, _>::new(model(cancel)),
        ShopPolicy { confirm: false },
    );
    let err = unconfirmed.run("Cancel 4711", &context).await.unwrap_err();
    assert!(
        matches!(err, ToolError::NotConfirmed { ref action } if action == "CancelOrder"),
        "{err}"
    );
    assert!(
        context.cancelled.lock().unwrap().is_empty(),
        "nothing changed"
    );

    let confirmed = ToolProgram::new(
        Program::<Decide, _>::new(model(cancel)),
        ShopPolicy { confirm: true },
    );
    let outcome = confirmed.run("Cancel 4711", &context).await.unwrap();
    assert_eq!(outcome.effect, Effect::Write);
    assert_eq!(*context.cancelled.lock().unwrap(), [4711]);
}

#[tokio::test]
async fn denied_failed_and_invalid_choices() {
    let find = r#"{"FindOrder": {"order_id": 4711}}"#;
    let assistant = ToolProgram::new(
        Program::<Decide, _>::new(model(find)),
        ShopPolicy { confirm: true },
    );
    let err = assistant
        .run("Where is 4711?", &shop("guest"))
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "FindOrder denied: guests cannot act on orders"
    );

    let missing = ToolProgram::new(
        Program::<Decide, _>::new(model(r#"{"FindOrder": {"order_id": 0}}"#)),
        ShopPolicy { confirm: true },
    );
    let err = missing
        .run("Where is 0?", &shop("alice"))
        .await
        .unwrap_err();
    assert!(matches!(err, ToolError::Execution { ref error, .. } if error == "no order 0"));

    let invalid = ToolProgram::new(
        Program::<Decide, _>::new(model(r#"{"DeleteEverything": {}}"#)).max_repairs(0),
        ShopPolicy { confirm: true },
    );
    assert!(matches!(
        invalid.run("x", &shop("alice")).await.unwrap_err(),
        ToolError::Choice(_)
    ));
}

#[tokio::test]
async fn tool_programs_offer_one_tool_per_variant() {
    let seen = Mutex::new(Vec::new());
    let provider = FnProvider::new(|request: &typedlm::Request| {
        seen.lock()
            .unwrap()
            .push((request.strategy, request.tools.len()));
        Ok(r#""Wait""#.into())
    })
    .capabilities(Capabilities::new(
        vec![Strategy::NativeSchema, Strategy::ToolCall],
        SchemaDialect::Generic,
    ));
    let assistant = ToolProgram::new(
        Program::<Decide, _>::new(&provider),
        ShopPolicy { confirm: false },
    );
    assistant.run("hello", &shop("alice")).await.unwrap();
    assert_eq!(*seen.lock().unwrap(), [(Strategy::ToolCall, 4)]);
}
