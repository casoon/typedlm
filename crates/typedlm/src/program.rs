//! Executing a signature against a provider, with repair.

use std::fmt;
use std::marker::PhantomData;
use std::time::Instant;

use serde_json::Value;
use tracing::Instrument;
use tracing::field::Empty;

use crate::compiled::CompiledProgram;
use crate::dialect::schema_for_dialect;
use crate::{
    Demonstration, Error, GenerationOptions, Provider, RepairTurn, Request, Signature, Strategy,
    Usage, output_schema, parse_output,
};

/// A signature bound to a provider: an executable, typed LLM call.
///
/// ```ignore
/// let classify = Program::<ClassifyTicket, _>::new(provider).temperature(0.0);
/// let out = classify.run("Production database is down").await?;
/// ```
pub struct Program<S, P> {
    provider: P,
    instructions: Option<String>,
    strategy: Option<Strategy>,
    options: GenerationOptions,
    max_repairs: usize,
    record_content: bool,
    demonstrations: Vec<Demonstration>,
    schema: Value,
    signature: PhantomData<fn() -> S>,
}

/// Shows the configuration, not the provider: providers may hold credentials.
impl<S: Signature, P> fmt::Debug for Program<S, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Program")
            .field("signature", &S::NAME)
            .field("instructions", &self.instructions)
            .field("demonstrations", &self.demonstrations.len())
            .field("strategy", &self.strategy)
            .field("options", &self.options)
            .field("max_repairs", &self.max_repairs)
            .field("record_content", &self.record_content)
            .finish_non_exhaustive()
    }
}

/// A successful run with what it cost.
#[derive(Debug, Clone)]
pub struct Execution<T> {
    pub output: T,
    /// Model calls made, including repairs.
    pub attempts: usize,
    /// Token usage summed over all attempts.
    pub usage: Usage,
    /// Model that produced the accepted answer, as reported by the provider.
    pub model: String,
    pub strategy: Strategy,
}

/// A failed run with what it cost — model calls are paid for even when they fail.
#[derive(Debug)]
pub struct Failed {
    pub error: Error,
    /// Model calls made, including repairs; 0 if none was made.
    pub attempts: usize,
    /// Token usage summed over all attempts.
    pub usage: Usage,
}

impl fmt::Display for Failed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for Failed {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl From<Failed> for Error {
    fn from(failed: Failed) -> Self {
        failed.error
    }
}

impl<S: Signature, P: Provider> Program<S, P> {
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            instructions: None,
            strategy: None,
            options: GenerationOptions::default(),
            max_repairs: 2,
            record_content: false,
            demonstrations: Vec::new(),
            schema: output_schema::<S>(),
            signature: PhantomData,
        }
    }

    /// Replaces the instructions generated from the signature's description.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Forces a strategy instead of the strongest one the provider supports.
    pub fn strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.options.temperature = Some(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.options.max_tokens = Some(max_tokens);
        self
    }

    /// How often an invalid answer is sent back with feedback (default 2). With 0,
    /// the first invalid answer's error is returned as is.
    pub fn max_repairs(mut self, max_repairs: usize) -> Self {
        self.max_repairs = max_repairs;
        self
    }

    /// The program's configuration as an artefact to review, store and load elsewhere
    /// (see [`CompiledProgram`]).
    pub fn compile(&self) -> CompiledProgram {
        CompiledProgram {
            format: crate::compiled::FORMAT,
            program: S::NAME.into(),
            signature: crate::compiled::signature_hash::<S>(),
            instructions: self.instructions.clone(),
            demonstrations: self.demonstrations.clone(),
            strategy: self.strategy,
            generation: self.options.clone(),
            max_repairs: self.max_repairs,
            provenance: None,
        }
    }

    #[cfg(feature = "optimize")]
    pub(crate) fn provider(&self) -> &P {
        &self.provider
    }

    /// Builds a program from a compiled configuration that was already checked.
    pub(crate) fn from_parts(provider: P, compiled: &CompiledProgram) -> Self {
        let mut program = Self::new(provider);
        program.instructions = compiled.instructions.clone();
        program.demonstrations = compiled.demonstrations.clone();
        program.strategy = compiled.strategy;
        program.options = compiled.generation.clone();
        program.max_repairs = compiled.max_repairs;
        program
    }

    /// Adds a worked example, sent before every input. A few well-chosen examples often
    /// help more than longer instructions, especially with small models.
    pub fn demonstration(mut self, input: impl Into<S>, output: &S::Output) -> Self
    where
        S::Output: serde::Serialize,
    {
        self.demonstrations.push(Demonstration {
            input: serde_json::to_value(input.into()).expect("inputs serialize to JSON"),
            output: serde_json::to_value(output).expect("outputs serialize to JSON"),
        });
        self
    }

    /// Records input and model answers in trace events (default off). They may
    /// contain personal data; enable only where traces are allowed to hold it.
    pub fn record_content(mut self, record: bool) -> Self {
        self.record_content = record;
        self
    }

    pub async fn run(&self, input: impl Into<S>) -> Result<S::Output, Error> {
        match self.execute(input).await {
            Ok(execution) => Ok(execution.output),
            Err(failed) => Err(failed.error),
        }
    }

    /// Like [`run`](Self::run), but reports attempts and token usage on success and
    /// on failure.
    ///
    /// Runs inside a `typedlm.execute` span (level INFO) with OpenTelemetry GenAI
    /// field names where they exist, and emits one DEBUG event per attempt. Input and
    /// answers are only recorded with [`record_content`](Self::record_content).
    pub async fn execute(&self, input: impl Into<S>) -> Result<Execution<S::Output>, Failed> {
        let span = tracing::info_span!(
            "typedlm.execute",
            typedlm.program = S::NAME,
            gen_ai.operation.name = "chat",
            typedlm.strategy = Empty,
            gen_ai.response.model = Empty,
            gen_ai.usage.input_tokens = Empty,
            gen_ai.usage.output_tokens = Empty,
            typedlm.attempts = Empty,
            typedlm.outcome = Empty,
            error.type = Empty,
        );
        let result = self
            .attempts(input.into(), &span)
            .instrument(span.clone())
            .await;

        let (attempts, usage) = match &result {
            Ok(e) => (e.attempts, e.usage),
            Err(f) => (f.attempts, f.usage),
        };
        span.record("typedlm.attempts", attempts);
        span.record("gen_ai.usage.input_tokens", usage.input_tokens);
        span.record("gen_ai.usage.output_tokens", usage.output_tokens);
        match &result {
            Ok(execution) => {
                span.record("gen_ai.response.model", execution.model.as_str());
                let outcome = if execution.attempts == 1 {
                    "valid"
                } else {
                    "repaired"
                };
                span.record("typedlm.outcome", outcome);
            }
            Err(failed) => {
                span.record("typedlm.outcome", "failed");
                span.record("error.type", failed.error.kind());
            }
        }
        result
    }

    async fn attempts(
        &self,
        input: S,
        span: &tracing::Span,
    ) -> Result<Execution<S::Output>, Failed> {
        let mut usage = Usage::default();
        let mut attempts = 0;
        let fail = |error, attempts, usage| Failed {
            error,
            attempts,
            usage,
        };

        let input = match serde_json::to_value(input) {
            Ok(input) => input,
            Err(e) => {
                let error = Error::InvalidInput {
                    message: e.to_string(),
                };
                return Err(fail(error, 0, usage));
            }
        };
        let capabilities = self.provider.capabilities();
        let strategy = self
            .strategy
            .or_else(|| capabilities.strategies.iter().min().copied())
            .unwrap_or(Strategy::PromptOnly);
        span.record("typedlm.strategy", strategy.as_str());
        let provider_schema = schema_for_dialect(&self.schema, capabilities.dialect);
        let schema_in_prompt = match strategy {
            Strategy::JsonMode | Strategy::PromptOnly => true,
            Strategy::NativeSchema => !capabilities.native_schema_visible,
            Strategy::ToolCall => false,
        };
        let instructions = self.build_instructions(schema_in_prompt, &provider_schema);
        if self.record_content {
            tracing::debug!(typedlm.input = %input, "typedlm input");
        }

        let mut repair = Vec::new();
        let mut failures = Vec::new();

        while attempts <= self.max_repairs {
            attempts += 1;
            let started = Instant::now();
            let response = self
                .provider
                .complete(Request {
                    instructions: instructions.clone(),
                    input: input.clone(),
                    output_schema: provider_schema.clone(),
                    strategy,
                    options: self.options.clone(),
                    demonstrations: self.demonstrations.clone(),
                    repair: repair.clone(),
                })
                .await;
            let latency_ms = started.elapsed().as_millis() as u64;
            let response = match response {
                Ok(response) => response,
                Err(e) => {
                    let error = Error::Provider(e);
                    tracing::debug!(
                        typedlm.attempt = attempts,
                        typedlm.latency_ms = latency_ms,
                        typedlm.result = error.kind(),
                        "typedlm attempt"
                    );
                    return Err(fail(error, attempts, usage));
                }
            };
            usage.input_tokens += response.usage.input_tokens;
            usage.output_tokens += response.usage.output_tokens;
            if self.record_content {
                tracing::debug!(
                    typedlm.attempt = attempts,
                    typedlm.response = response.content.as_str(),
                    "typedlm response"
                );
            }

            let parsed = parse_output::<S>(&response, &self.schema);
            tracing::debug!(
                typedlm.attempt = attempts,
                typedlm.latency_ms = latency_ms,
                gen_ai.usage.input_tokens = response.usage.input_tokens,
                gen_ai.usage.output_tokens = response.usage.output_tokens,
                typedlm.result = parsed.as_ref().map_or_else(Error::kind, |_| "ok"),
                "typedlm attempt"
            );
            match parsed {
                Ok(output) => {
                    return Ok(Execution {
                        output,
                        attempts,
                        usage,
                        model: response.model,
                        strategy,
                    });
                }
                Err(error) if error.is_repairable() && self.max_repairs > 0 => {
                    if let Some(feedback) = error.repair_feedback() {
                        repair.push(RepairTurn {
                            response: response.content,
                            feedback,
                        });
                    }
                    failures.push(error);
                }
                Err(error) => return Err(fail(error, attempts, usage)),
            }
        }
        let error = Error::RetryLimitExceeded { attempts: failures };
        Err(fail(error, attempts, usage))
    }

    /// The instructions sent to the model, without the schema: the custom ones or those
    /// generated from the signature's description.
    pub(crate) fn effective_instructions(&self) -> String {
        match &self.instructions {
            Some(custom) => custom.clone(),
            None => format!(
                "{}\n\nThe input is given as JSON. Respond with a single JSON value that \
                 matches the output schema.",
                S::DESCRIPTION
            ),
        }
    }

    fn build_instructions(&self, schema_in_prompt: bool, schema: &Value) -> String {
        let mut text = self.effective_instructions();
        if schema_in_prompt {
            text.push_str("\n\nOutput JSON Schema:\n");
            text.push_str(&schema.to_string());
        }
        text
    }
}
