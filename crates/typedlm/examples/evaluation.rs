//! Measure a classifier on a labelled dataset.
//!
//! ```sh
//! TYPEDLM_MODEL=qwen3:32b cargo run --example evaluation
//! ```
//!
//! `examples/data/tickets.jsonl` labels only some fields per ticket; each line is
//! checked against the output type when loading. Keep datasets much larger than
//! this one: the report's confidence interval shows how little 8 examples say.

mod common;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::eval::{Dataset, ExactMatch, evaluate};
use typedlm::prelude::*;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
enum Urgency {
    Low,
    Medium,
    High,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
enum Category {
    Billing,
    Technical,
    Shipping,
    /// Anything that fits none of the other categories.
    Other,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct TicketClassification {
    urgency: Urgency,
    category: Category,
}

/// Classify a customer support ticket by urgency and category.
#[derive(TypedLm, Serialize, Deserialize, Clone, Debug)]
#[lm(output = TicketClassification)]
struct ClassifyTicket {
    /// The ticket as written by the customer.
    text: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data/tickets.jsonl");
    let dataset = Dataset::<ClassifyTicket>::from_jsonl(path)?;

    let classify = Program::<ClassifyTicket, _>::new(common::provider()).temperature(0.0);
    let report = evaluate(&classify, &dataset, &ExactMatch, 4).await;

    println!("{report}");
    for (index, error) in &report.failures {
        println!("example {index} failed: {error}");
    }
    // The report is plain data, e.g. for storing next to the dataset.
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
