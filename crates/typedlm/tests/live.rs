//! Smoke tests against a real OpenAI-compatible endpoint. Ignored by default; run with
//!
//! ```sh
//! TYPEDLM_LIVE_BASE_URL=http://localhost:11434/v1 TYPEDLM_LIVE_MODEL=qwen3:32b \
//! TYPEDLM_LIVE_DIALECT=ollama cargo test --test live -- --ignored --nocapture
//! ```
//!
//! `TYPEDLM_LIVE_API_KEY` is optional; `TYPEDLM_LIVE_DIALECT` is `strict` (default),
//! `generic`, or `ollama` (generic, schema also in the prompt); `TYPEDLM_LIVE_STRATEGIES` limits strategies (comma-separated, e.g.
//! `native_schema,json_mode`).
#![cfg(all(feature = "http", feature = "eval"))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::eval::{Dataset, ExactMatch, evaluate};
use typedlm::http::OpenAiCompatible;
use typedlm::prelude::*;
use typedlm::{Capabilities, SchemaDialect, Strategy};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
enum Urgency {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
enum Category {
    Billing,
    Technical,
    Shipping,
    Other,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Classification {
    urgency: Urgency,
    category: Category,
}

/// Classify a customer support ticket by urgency and category.
#[derive(TypedLm, Serialize, Deserialize, Clone, Debug)]
#[lm(output = Classification)]
struct ClassifyTicket {
    text: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct LineItem {
    description: String,
    quantity: u32,
    /// Unit price in cents.
    unit_price_cents: u64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Invoice {
    number: String,
    /// Issue date as YYYY-MM-DD.
    date: String,
    items: Vec<LineItem>,
    /// Grand total in cents, including tax.
    total_cents: u64,
    note: Option<String>,
}

/// Extract the invoice data from the text.
#[derive(TypedLm, Serialize)]
#[lm(output = Invoice)]
struct ExtractInvoice {
    text: String,
}

const INVOICE: &str = "Invoice no. RE-2026-0042, issued 14 September 2026.\n\
    2 x Server maintenance at 150.00 EUR each\n\
    1 x Emergency call-out at 89.50 EUR\n\
    Subtotal 389.50 EUR, VAT 19 % 74.01 EUR, total 463.51 EUR.";

const TICKETS: &str = r#"
{"input": {"text": "Our production database has been down for an hour, nobody can log in."}, "expected": {"urgency": "High", "category": "Technical"}}
{"input": {"text": "I was charged twice for my subscription this month."}, "expected": {"category": "Billing"}}
{"input": {"text": "My parcel shows delivered but it never arrived."}, "expected": {"category": "Shipping"}}
{"input": {"text": "How do I change my profile picture?"}, "expected": {"urgency": "Low"}}
{"input": {"text": "Can you send me a copy of last year's invoice?"}, "expected": {"category": "Billing"}}
"#;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn provider() -> OpenAiCompatible {
    let base_url = env("TYPEDLM_LIVE_BASE_URL").expect("set TYPEDLM_LIVE_BASE_URL");
    let model = env("TYPEDLM_LIVE_MODEL").expect("set TYPEDLM_LIVE_MODEL");
    let capabilities = match env("TYPEDLM_LIVE_DIALECT").as_deref() {
        Some("generic") => Capabilities::new(strategies(), SchemaDialect::Generic),
        Some("ollama") => {
            Capabilities::new(strategies(), SchemaDialect::Generic).native_schema_visible(false)
        }
        _ => Capabilities::new(strategies(), SchemaDialect::OpenAiStrict),
    };
    let mut provider = OpenAiCompatible::new(base_url, model).capabilities(capabilities);
    if let Some(key) = env("TYPEDLM_LIVE_API_KEY") {
        provider = provider.api_key(key);
    }
    provider
}

fn strategies() -> Vec<Strategy> {
    let all = [
        Strategy::NativeSchema,
        Strategy::ToolCall,
        Strategy::JsonMode,
        Strategy::PromptOnly,
    ];
    match env("TYPEDLM_LIVE_STRATEGIES") {
        Some(list) => all
            .into_iter()
            .filter(|s| list.split(',').any(|n| n.trim() == s.as_str()))
            .collect(),
        None => all.to_vec(),
    }
}

#[tokio::test]
#[ignore = "needs a live endpoint"]
async fn every_strategy_classifies_and_extracts() {
    let mut failures = Vec::new();
    for strategy in strategies() {
        let classify = Program::<ClassifyTicket, _>::new(provider())
            .strategy(strategy)
            .temperature(0.0);
        match classify
            .execute("The payment page returns error 500 for all customers")
            .await
        {
            Ok(e) => println!(
                "{:<13} classify  {:?} in {} attempt(s), {} tokens",
                strategy.as_str(),
                e.output,
                e.attempts,
                e.usage.input_tokens + e.usage.output_tokens
            ),
            Err(f) => failures.push(format!(
                "{} classify: {} ({:?})",
                strategy.as_str(),
                f,
                f.error
            )),
        }

        let extract = Program::<ExtractInvoice, _>::new(provider())
            .strategy(strategy)
            .temperature(0.0);
        match extract.execute(INVOICE).await {
            Ok(e) => {
                println!(
                    "{:<13} extract   {} items, total {} cents, date {}, {} attempt(s)",
                    strategy.as_str(),
                    e.output.items.len(),
                    e.output.total_cents,
                    e.output.date,
                    e.attempts
                );
                let iso_date = e.output.date == "2026-09-14";
                if e.output.total_cents != 46351 || e.output.items.len() != 2 || !iso_date {
                    failures.push(format!(
                        "{} extract: wrong values {:?}",
                        strategy.as_str(),
                        e.output
                    ));
                }
            }
            Err(f) => failures.push(format!(
                "{} extract: {} ({:?})",
                strategy.as_str(),
                f,
                f.error
            )),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[tokio::test]
#[ignore = "needs a live endpoint"]
async fn small_evaluation() {
    let dataset = Dataset::<ClassifyTicket>::from_jsonl_str(TICKETS).unwrap();
    let program = Program::<ClassifyTicket, _>::new(provider()).temperature(0.0);
    let report = evaluate(&program, &dataset, &ExactMatch, 2).await;
    println!("{report}");
    for (index, error) in &report.failures {
        println!("failure #{index}: {error}");
    }
    assert_eq!(report.failed, 0);
}

/// What to do with a support message.
#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
enum SupportAction {
    /// Look up the status of an order by its number.
    FindOrder { order_id: u64 },
    /// Cancel an order by its number, with the customer's reason.
    CancelOrder { order_id: u64, reason: String },
    /// Hand the conversation to a human agent.
    Escalate,
}

/// Decide the single next action for a customer support message.
#[derive(TypedLm, Serialize)]
#[lm(output = SupportAction)]
struct DecideAction {
    message: String,
}

#[tokio::test]
#[ignore = "needs a live endpoint"]
async fn native_tools_choose_actions() {
    let program = Program::<DecideAction, _>::new(provider())
        .strategy(Strategy::ToolCall)
        .temperature(0.0);
    let cases = [
        ("Where is my order 4711? It has not arrived.", "FindOrder"),
        (
            "Please cancel order 8812, I ordered the wrong size.",
            "CancelOrder",
        ),
        ("I want to talk to a real person now.", "Escalate"),
    ];
    let mut failures = Vec::new();
    for (message, expected) in cases {
        match program.execute(message).await {
            Ok(e) => {
                let name = serde_json::to_value(&e.output).unwrap();
                let name = name
                    .as_str()
                    .map(String::from)
                    .or_else(|| name.as_object().and_then(|m| m.keys().next().cloned()))
                    .unwrap();
                println!(
                    "{expected:<12} → {:?} in {} attempt(s)",
                    e.output, e.attempts
                );
                if name != expected {
                    failures.push(format!("{message}: expected {expected}, got {name}"));
                }
            }
            Err(f) => failures.push(format!("{message}: {f} ({:?})", f.error)),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
