//! Typed tools: the model chooses one action, Rust checks and executes it.
//!
//! The actions are the variants of an enum, used as a signature's output type. Choosing an
//! action is then an ordinary program run — validated, repaired, evaluated and optimized like
//! any other. [`ToolProgram`] adds what happens after the choice: authorization and, for
//! actions that change data, confirmation — both by your [`Policy`] — before your code
//! executes the action.
//!
//! ```ignore
//! /// What the assistant does next.
//! #[derive(Serialize, Deserialize, JsonSchema)]
//! enum SupportAction {
//!     /// Look up an order.
//!     FindOrder { order_id: u64 },
//!     /// Cancel an order.
//!     CancelOrder { order_id: u64, reason: String },
//! }
//!
//! impl Action for SupportAction {
//!     type Context = Session;
//!     type Output = String;
//!     type Error = ShopError;
//!
//!     fn effect(&self) -> Effect {
//!         match self {
//!             Self::FindOrder { .. } => Effect::Read,
//!             Self::CancelOrder { .. } => Effect::Write,
//!         }
//!     }
//!
//!     async fn execute(self, session: &Session) -> Result<String, ShopError> {
//!         match self {
//!             Self::FindOrder { order_id } => session.shop.order(order_id).await,
//!             Self::CancelOrder { order_id, reason } => session.shop.cancel(order_id, &reason).await,
//!         }
//!     }
//! }
//!
//! let assistant = ToolProgram::new(program, MyPolicy);
//! let outcome = assistant.run("Where is order 4711?", &session).await?;
//! ```
//!
//! One action per call: the result goes back to your code, not to the model. There is no
//! loop of tool calls.

use std::fmt;
use std::future::Future;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tracing::Instrument;

use crate::{Error, Program, Provider, Signature, Strategy};

/// Whether an action only reads or changes something.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// No side effects; runs after authorization.
    Read,
    /// Changes data or the outside world; runs only after authorization and confirmation.
    Write,
}

/// An action the model can choose: implemented by the enum used as a signature's output.
pub trait Action: Serialize + DeserializeOwned {
    /// What execution needs, e.g. a session with the user's identity and clients.
    type Context;
    type Output;
    type Error;

    fn effect(&self) -> Effect;

    fn execute(
        self,
        context: &Self::Context,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>>;
}

/// Decides whether a chosen action may run. Both checks see the action with its validated
/// arguments and the context.
pub trait Policy<A: Action> {
    /// Authorization, for every action. `Err` carries the reason.
    fn authorize(&self, action: &A, context: &A::Context) -> Result<(), String>;

    /// Confirmation for [`Effect::Write`] actions, e.g. by asking the user. The default
    /// refuses: writes run only with a policy that confirms them.
    fn confirm(&self, action: &A, context: &A::Context) -> impl Future<Output = bool> {
        let _ = (action, context);
        async { false }
    }
}

/// A chosen and executed action.
#[derive(Debug)]
pub struct Outcome<T> {
    /// Variant name of the action, e.g. `FindOrder`.
    pub action: String,
    pub effect: Effect,
    pub output: T,
}

/// Why no action was executed, or why it failed.
#[derive(Debug)]
pub enum ToolError<E> {
    /// The model gave no valid choice.
    Choice(Error),
    /// The policy refused the action.
    Denied { action: String, reason: String },
    /// A write action was not confirmed.
    NotConfirmed { action: String },
    /// The action ran and failed.
    Execution { action: String, error: E },
}

impl<E: fmt::Display> fmt::Display for ToolError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Choice(error) => write!(f, "no valid action: {error}"),
            Self::Denied { action, reason } => write!(f, "{action} denied: {reason}"),
            Self::NotConfirmed { action } => {
                write!(f, "{action} changes data and was not confirmed")
            }
            Self::Execution { action, error } => write!(f, "{action} failed: {error}"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for ToolError<E> {}

/// A program whose output is an [`Action`], with the policy that guards execution.
pub struct ToolProgram<S, P, Po> {
    program: Program<S, P>,
    policy: Po,
}

impl<S, P, Po> ToolProgram<S, P, Po>
where
    S: Signature,
    S::Output: Action,
    P: Provider,
    Po: Policy<S::Output>,
{
    /// Uses native tool calls when the provider supports them: each variant becomes a tool
    /// with its doc comment as description.
    pub fn new(program: Program<S, P>, policy: Po) -> Self {
        let program = if program.provider_supports(Strategy::ToolCall) {
            program.strategy(Strategy::ToolCall)
        } else {
            program
        };
        Self { program, policy }
    }

    pub fn program(&self) -> &Program<S, P> {
        &self.program
    }

    /// Chooses an action for `input`, checks it against the policy and executes it.
    pub async fn run(
        &self,
        input: impl Into<S>,
        context: &<S::Output as Action>::Context,
    ) -> Result<Outcome<<S::Output as Action>::Output>, ToolError<<S::Output as Action>::Error>>
    {
        let action = self.program.run(input).await.map_err(ToolError::Choice)?;
        let name = action_name(&action);
        let effect = action.effect();
        let span = tracing::info_span!(
            "typedlm.tool",
            typedlm.program = S::NAME,
            gen_ai.tool.name = name.as_str(),
            typedlm.effect = if effect == Effect::Read {
                "read"
            } else {
                "write"
            },
            typedlm.outcome = tracing::field::Empty,
        );
        let result = async {
            if let Err(reason) = self.policy.authorize(&action, context) {
                return Err(ToolError::Denied {
                    action: name.clone(),
                    reason,
                });
            }
            if effect == Effect::Write && !self.policy.confirm(&action, context).await {
                return Err(ToolError::NotConfirmed {
                    action: name.clone(),
                });
            }
            match action.execute(context).await {
                Ok(output) => Ok(Outcome {
                    action: name.clone(),
                    effect,
                    output,
                }),
                Err(error) => Err(ToolError::Execution {
                    action: name.clone(),
                    error,
                }),
            }
        }
        .instrument(span.clone())
        .await;
        span.record(
            "typedlm.outcome",
            match &result {
                Ok(_) => "executed",
                Err(ToolError::Denied { .. }) => "denied",
                Err(ToolError::NotConfirmed { .. }) => "not_confirmed",
                Err(ToolError::Execution { .. }) => "failed",
                Err(ToolError::Choice(_)) => "no_choice",
            },
        );
        result
    }
}

/// The variant name: the key of an externally tagged variant, or the string of a unit one.
fn action_name<A: Serialize>(action: &A) -> String {
    match serde_json::to_value(action) {
        Ok(Value::String(name)) => name,
        Ok(Value::Object(map)) if map.len() == 1 => map.keys().next().cloned().unwrap_or_default(),
        _ => "action".into(),
    }
}

/// A tool offered to the model: one variant of an action enum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema of the arguments, always an object.
    pub parameters: Value,
    /// A variant without fields; its value is the bare name.
    pub unit: bool,
}

impl ToolSpec {
    /// The output value for a call of this tool, in serde's externally tagged form.
    pub fn output(&self, arguments: Value) -> Value {
        if self.unit {
            Value::String(self.name.clone())
        } else {
            let mut map = Map::new();
            map.insert(self.name.clone(), arguments);
            Value::Object(map)
        }
    }
}

/// Splits the schema of an externally tagged enum into one tool per variant. `None` when the
/// schema is no such enum, or a variant has non-object fields (e.g. a tuple variant).
pub fn tool_specs(schema: &Value) -> Option<Vec<ToolSpec>> {
    let branches = schema
        .get("oneOf")
        .or_else(|| schema.get("anyOf"))?
        .as_array()?;
    let defs = schema.get("$defs").cloned();
    let empty = json!({ "type": "object", "properties": {}, "required": [], "additionalProperties": false });
    let mut specs = Vec::new();
    for branch in branches {
        let description = branch
            .get("description")
            .and_then(Value::as_str)
            .map(String::from);
        match branch.get("type").and_then(Value::as_str) {
            Some("string") => {
                let names: Vec<&str> = match (branch.get("const"), branch.get("enum")) {
                    (Some(Value::String(name)), _) => vec![name.as_str()],
                    (_, Some(Value::Array(values))) => {
                        values.iter().filter_map(Value::as_str).collect()
                    }
                    _ => return None,
                };
                for name in names {
                    specs.push(ToolSpec {
                        name: name.into(),
                        description: description.clone(),
                        parameters: empty.clone(),
                        unit: true,
                    });
                }
            }
            Some("object") => {
                let properties = branch.get("properties")?.as_object()?;
                if properties.len() != 1 {
                    return None;
                }
                let (name, fields) = properties.iter().next()?;
                let mut parameters = resolve(fields, schema).clone();
                if parameters.get("type").and_then(Value::as_str) != Some("object") {
                    return None;
                }
                if let (Some(defs), Value::Object(map)) = (&defs, &mut parameters) {
                    map.insert("$defs".into(), defs.clone());
                }
                specs.push(ToolSpec {
                    name: name.clone(),
                    description,
                    parameters,
                    unit: false,
                });
            }
            _ => return None,
        }
    }
    (!specs.is_empty()).then_some(specs)
}

/// Follows a local `$ref` once.
fn resolve<'a>(schema: &'a Value, root: &'a Value) -> &'a Value {
    schema
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|r| r.strip_prefix('#'))
        .and_then(|p| root.pointer(p))
        .unwrap_or(schema)
}
