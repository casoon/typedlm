//! Classify support tickets into enums.
//!
//! ```sh
//! TYPEDLM_MODEL=qwen3:32b cargo run --example classification
//! ```

mod common;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
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
#[derive(TypedLm, Serialize)]
#[lm(output = TicketClassification)]
struct ClassifyTicket {
    /// The ticket as written by the customer.
    text: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let classify = Program::<ClassifyTicket, _>::new(common::provider()).temperature(0.0);

    // `run` returns the typed output; a single `String` field lets it take text.
    let ticket = classify
        .run("Production database is down since 9:00")
        .await?;
    match ticket.urgency {
        Urgency::High => println!("escalate: {ticket:?}"),
        Urgency::Medium | Urgency::Low => println!("queue: {ticket:?}"),
    }

    // `execute` also reports what the call cost.
    let execution = classify.execute("I was charged twice this month").await?;
    println!(
        "{:?} — {} attempt(s), {} tokens, model {}",
        execution.output,
        execution.attempts,
        execution.usage.input_tokens + execution.usage.output_tokens,
        execution.model
    );
    Ok(())
}
