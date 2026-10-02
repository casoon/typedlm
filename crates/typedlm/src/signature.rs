use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// The typed contract of one LLM call.
///
/// The implementing type is the input; `Output` is what the model's answer must
/// deserialize into. Usually derived with `#[derive(TypedLm)]`, but implementing it
/// by hand is supported.
pub trait Signature: Serialize {
    type Output: DeserializeOwned + JsonSchema;

    /// Stable name of the program, used in traces and artifacts.
    const NAME: &'static str;

    /// Task description the default instructions are built from.
    const DESCRIPTION: &'static str;

    /// Domain checks beyond what the schema expresses. Each returned message is
    /// sent back to the model when repairing, so phrase it as an instruction.
    fn validate(_output: &Self::Output) -> Result<(), Vec<String>> {
        Ok(())
    }
}
