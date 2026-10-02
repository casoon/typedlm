---
title: Tools
description: Let the model choose one typed action; Rust authorizes, confirms writes, and executes it.
order: 9
---

A tool in TypedLM is a variant of an enum. The model chooses one variant with its arguments —
validated like any output — and your code decides whether it runs. *The model proposes, Rust
validates and executes.*

## Actions

```rust
/// What the assistant does next.
#[derive(Serialize, Deserialize, JsonSchema)]
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
```

The doc comments matter: each becomes the description of its tool. Variants with named fields and
variants without fields work; tuple variants do not.

Choosing an action is an ordinary program run, so validation, repair, evaluation, optimization and
compiled programs apply unchanged — measure how often the right tool is chosen like any other
classification.

## Executing

Implement `Action` on the enum: which variants change data, and what each one does.

```rust
use typedlm::tools::{Action, Effect};

impl Action for SupportAction {
    type Context = Session;
    type Output = String;
    type Error = ShopError;

    fn effect(&self) -> Effect {
        match self {
            Self::CancelOrder { .. } => Effect::Write,
            _ => Effect::Read,
        }
    }

    async fn execute(self, session: &Session) -> Result<String, ShopError> {
        match self {
            Self::FindOrder { order_id } => session.shop.status(order_id).await,
            Self::CancelOrder { order_id, reason } => session.shop.cancel(order_id, &reason).await,
            Self::Escalate => session.handover().await,
        }
    }
}
```

The `match` is exhaustive: a new variant does not compile until it has an implementation.

## Policy

A `Policy` decides whether a chosen action may run:

```rust
use typedlm::tools::Policy;

struct SupportPolicy;

impl Policy<SupportAction> for SupportPolicy {
    fn authorize(&self, action: &SupportAction, session: &Session) -> Result<(), String> {
        match action {
            SupportAction::CancelOrder { order_id, .. } if !session.owns(*order_id) => {
                Err("the order belongs to another customer".into())
            }
            _ => Ok(()),
        }
    }

    async fn confirm(&self, action: &SupportAction, session: &Session) -> bool {
        session.ask_user(&format!("{action:?}?")).await
    }
}
```

- `authorize` runs for every action, with the validated arguments and your context — the user's
  identity, permissions.
- `confirm` runs for `Effect::Write` actions only. **Its default refuses**: an action that changes
  data runs only with a policy that confirms it.

## Running

```rust
use typedlm::tools::ToolProgram;

let assistant = ToolProgram::new(Program::new(provider), SupportPolicy);
let outcome = assistant.run("Please cancel order 8812, wrong size.", &session).await?;
println!("{}: {}", outcome.action, outcome.output);
```

`run` chooses an action, authorizes it, confirms it if it writes, and executes it. Errors say which
step stopped it: `Choice` (no valid action), `Denied`, `NotConfirmed`, `Execution`. A
`typedlm.tool` tracing span records the action, its effect and the outcome.

`ToolProgram` uses native tool calls when the provider supports them: every variant is offered as a
tool with its description, and the model must call one. Tool calls a local model writes as text are
recognised too. Tested live with `qwen3:32b` on Ollama.

## One action per call

The result goes back to your code, not to the model; there is no loop of tool calls. Ask again with
the result in the input if a second step is needed — every step stays visible and checked in your
code. Multi-step agents are out of scope.
