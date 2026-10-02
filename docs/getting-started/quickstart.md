---
title: Quickstart
description: Define a contract, bind it to a model and call it.
order: 2
---

## Describe the call

The input struct carries the derive; the output is a separate, ordinary type. The doc comment on
the input becomes the instruction, doc comments on fields end up in the schema.

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::prelude::*;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
enum Urgency { Low, Medium, High }

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
```

Give classifications a way out such as `Other` or an `Option`; without it the model has to guess.

## Bind it to a model

```rust
use typedlm::http::OpenAiCompatible;

// A local Ollama …
let provider = OpenAiCompatible::ollama("qwen3:32b");
// … or OpenAI.
let provider = OpenAiCompatible::new("https://api.openai.com/v1", "gpt-5-mini").api_key(key);

let classify = Program::<ClassifyTicket, _>::new(provider).temperature(0.0);
```

## Call it

```rust
let ticket = classify.run("Production database is down since 9:00").await?;
println!("{:?} / {:?}", ticket.urgency, ticket.category);
```

`run` takes the input struct, or a string when the struct has exactly one `String` field. It
returns the output type or a `typedlm::Error`. `execute` returns the same output together with the
number of attempts, token usage, the model that answered and the strategy used — on failure too.

## Run the examples

The repository contains three runnable examples in `crates/typedlm/examples/`:

```sh
TYPEDLM_MODEL=qwen3:32b cargo run --example classification
TYPEDLM_MODEL=qwen3:32b cargo run --example extraction
TYPEDLM_MODEL=qwen3:32b cargo run --example evaluation
```

Without `TYPEDLM_API_KEY` they use a local Ollama; with it, the OpenAI API or `TYPEDLM_BASE_URL`.
