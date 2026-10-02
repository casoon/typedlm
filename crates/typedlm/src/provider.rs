use std::fmt;
use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

/// Executes requests against a language model.
///
/// Transport concerns (HTTP retries, rate limits, backoff) belong to the provider;
/// repairing invalid answers is the caller's job.
pub trait Provider: Send + Sync {
    fn capabilities(&self) -> Capabilities;

    fn complete(
        &self,
        request: Request,
    ) -> impl Future<Output = Result<Response, ProviderError>> + Send;
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Object-safe counterpart of [`Provider`], for holding different providers in
/// one collection (e.g. comparing models). Implemented for every `Provider`;
/// `Box<dyn DynProvider>` is itself a `Provider`.
pub trait DynProvider: Send + Sync {
    fn capabilities(&self) -> Capabilities;

    fn complete_boxed(&self, request: Request) -> BoxFuture<'_, Result<Response, ProviderError>>;
}

impl<P: Provider> DynProvider for P {
    fn capabilities(&self) -> Capabilities {
        Provider::capabilities(self)
    }

    fn complete_boxed(&self, request: Request) -> BoxFuture<'_, Result<Response, ProviderError>> {
        Box::pin(self.complete(request))
    }
}

/// A borrowed provider, so a test can inspect it after handing it to a program.
impl<P: Provider> Provider for &P {
    fn capabilities(&self) -> Capabilities {
        (**self).capabilities()
    }

    fn complete(
        &self,
        request: Request,
    ) -> impl Future<Output = Result<Response, ProviderError>> + Send {
        (**self).complete(request)
    }
}

/// A shared provider, e.g. one client for several programs.
impl<P: Provider> Provider for std::sync::Arc<P> {
    fn capabilities(&self) -> Capabilities {
        (**self).capabilities()
    }

    fn complete(
        &self,
        request: Request,
    ) -> impl Future<Output = Result<Response, ProviderError>> + Send {
        (**self).complete(request)
    }
}

impl Provider for Box<dyn DynProvider> {
    fn capabilities(&self) -> Capabilities {
        (**self).capabilities()
    }

    fn complete(
        &self,
        request: Request,
    ) -> impl Future<Output = Result<Response, ProviderError>> + Send {
        (**self).complete_boxed(request)
    }
}

/// What a provider supports for getting structured output.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Capabilities {
    /// Supported strategies; the strongest one is chosen unless overridden.
    pub strategies: Vec<Strategy>,
    /// JSON Schema flavour the provider accepts.
    pub dialect: SchemaDialect,
    /// Whether the model itself sees the schema under [`Strategy::NativeSchema`].
    /// Servers that only use it to constrain decoding (e.g. Ollama) hide field
    /// descriptions from the model; then the schema is also put into the prompt.
    pub native_schema_visible: bool,
}

impl Capabilities {
    /// Capabilities with a visible native schema.
    pub fn new(strategies: Vec<Strategy>, dialect: SchemaDialect) -> Self {
        Self {
            strategies,
            dialect,
            native_schema_visible: true,
        }
    }

    pub fn native_schema_visible(mut self, visible: bool) -> Self {
        self.native_schema_visible = visible;
        self
    }
}

/// How structured output is obtained, strongest first.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Strategy {
    /// The provider enforces the schema while decoding.
    NativeSchema,
    /// The schema is passed as the parameters of a forced tool call.
    ToolCall,
    /// The provider guarantees syntactically valid JSON, not the schema.
    JsonMode,
    /// The schema is only described in the prompt.
    PromptOnly,
}

impl Strategy {
    /// Stable lowercase name, as used in traces.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NativeSchema => "native_schema",
            Self::ToolCall => "tool_call",
            Self::JsonMode => "json_mode",
            Self::PromptOnly => "prompt_only",
        }
    }
}

/// JSON Schema flavour accepted by a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SchemaDialect {
    /// Strict structured outputs of OpenAI-compatible APIs: every property
    /// required, `additionalProperties: false`, restricted keywords.
    OpenAiStrict,
    /// Plain JSON Schema as generated.
    Generic,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Request {
    /// System instructions, complete: they already describe the schema when the
    /// strategy does not transport it.
    pub instructions: String,
    pub input: Value,
    /// Output schema in the provider's dialect.
    pub output_schema: Value,
    pub strategy: Strategy,
    pub options: GenerationOptions,
    /// Worked examples, sent before the input as user/assistant turns.
    #[serde(default)]
    pub demonstrations: Vec<Demonstration>,
    /// Earlier answers and the feedback on them, oldest first; empty on the first
    /// attempt. Providers send each as an assistant turn followed by a user turn.
    pub repair: Vec<RepairTurn>,
}

/// A worked example: an input and the output the model should give for it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Demonstration {
    pub input: Value,
    pub output: Value,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RepairTurn {
    pub response: String,
    pub feedback: String,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GenerationOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Response {
    /// Raw answer text, before parsing and validation.
    pub content: String,
    /// Model that actually answered, as reported by the provider.
    pub model: String,
    pub finish: FinishReason,
    pub usage: Usage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    /// Output token limit reached; the answer is truncated.
    Length,
    /// The model declined to answer.
    Refusal,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum ProviderError {
    /// Network or protocol failure; retries are exhausted.
    Transport(String),
    /// The provider answered with an error status.
    Status { code: u16, body: String },
    /// The provider's answer could not be read.
    InvalidResponse(String),
    /// Recording or replaying interactions failed (see `typedlm::testing`).
    Recording(String),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(msg) => write!(f, "transport error: {msg}"),
            Self::Status { code, body } => write!(f, "provider returned status {code}: {body}"),
            Self::InvalidResponse(msg) => write!(f, "invalid provider response: {msg}"),
            Self::Recording(msg) => write!(f, "recording: {msg}"),
        }
    }
}

impl std::error::Error for ProviderError {}
