//! Typed, testable LLM programs for Rust.
//!
//! A [`Signature`] is the typed contract of one LLM call: the implementing type is
//! the input, [`Signature::Output`] is an ordinary Rust type the model's answer is
//! validated against. A [`Provider`] executes requests against a model.

mod dialect;
mod error;
#[cfg(feature = "eval")]
pub mod eval;
#[cfg(feature = "http")]
pub mod http;
mod json;
mod output;
mod program;
mod provider;
mod schema;
mod signature;

pub use dialect::schema_for_dialect;
pub use error::Error;
pub use output::{output_schema, parse_output};
pub use program::{Execution, Failed, Program};
pub use schema::Violation;

pub use provider::{
    Capabilities, DynProvider, FinishReason, GenerationOptions, Provider, ProviderError,
    RepairTurn, Request, Response, SchemaDialect, Strategy, Usage,
};
pub use signature::Signature;

#[cfg(feature = "derive")]
pub use typedlm_macros::TypedLm;

pub mod prelude {
    pub use crate::{Error, Program, Provider, Signature};

    #[cfg(feature = "derive")]
    pub use crate::TypedLm;
}
