//! Provider for OpenAI-compatible chat completion APIs (feature `http`).
//!
//! Covers OpenAI itself and every service speaking the same protocol — local
//! servers such as Ollama, and routers that translate to other vendors.

use std::time::Duration;

use serde_json::{Value, json};

use crate::{
    Capabilities, FinishReason, Provider, ProviderError, Request, Response, SchemaDialect,
    Strategy, Usage,
};

const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

const ALL_STRATEGIES: [Strategy; 4] = [
    Strategy::NativeSchema,
    Strategy::ToolCall,
    Strategy::JsonMode,
    Strategy::PromptOnly,
];

#[derive(Debug, Clone)]
pub struct OpenAiCompatible {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
    capabilities: Capabilities,
    max_transport_retries: u32,
    initial_backoff: Duration,
}

impl OpenAiCompatible {
    /// `base_url` is the API root, e.g. `https://api.openai.com/v1` or
    /// `http://localhost:11434/v1`; `/chat/completions` is appended.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            api_key: None,
            capabilities: Capabilities::new(ALL_STRATEGIES.to_vec(), SchemaDialect::OpenAiStrict),
            max_transport_retries: 2,
            initial_backoff: Duration::from_millis(500),
        }
    }

    /// A local Ollama server at `http://localhost:11434/v1`: generic schema dialect,
    /// and the schema is also sent in the prompt, since Ollama only uses it to
    /// constrain decoding.
    pub fn ollama(model: impl Into<String>) -> Self {
        Self::new("http://localhost:11434/v1", model).capabilities(
            Capabilities::new(ALL_STRATEGIES.to_vec(), SchemaDialect::Generic)
                .native_schema_visible(false),
        )
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Uses a preconfigured client (timeouts, proxies, TLS settings).
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Declares what the endpoint supports; defaults to everything with the strict
    /// OpenAI schema dialect. Servers without strict structured outputs should
    /// declare [`SchemaDialect::Generic`].
    pub fn capabilities(mut self, capabilities: Capabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Retries after connection errors, 429 and 5xx responses (default 2).
    pub fn max_transport_retries(mut self, retries: u32) -> Self {
        self.max_transport_retries = retries;
        self
    }

    /// First backoff delay, doubled per retry, unless the server sends `Retry-After`.
    pub fn initial_backoff(mut self, backoff: Duration) -> Self {
        self.initial_backoff = backoff;
        self
    }

    fn body(&self, request: &Request) -> Value {
        let mut messages = vec![json!({ "role": "system", "content": request.instructions })];
        for demo in &request.demonstrations {
            messages.push(json!({ "role": "user", "content": demo.input.to_string() }));
            messages.push(json!({ "role": "assistant", "content": demo.output.to_string() }));
        }
        messages.push(json!({ "role": "user", "content": request.input.to_string() }));
        for turn in &request.repair {
            messages.push(json!({ "role": "assistant", "content": turn.response }));
            messages.push(json!({ "role": "user", "content": turn.feedback }));
        }

        let mut body = json!({ "model": self.model, "messages": messages });
        let strict = self.capabilities.dialect == SchemaDialect::OpenAiStrict;
        match request.strategy {
            Strategy::NativeSchema => {
                body["response_format"] = json!({
                    "type": "json_schema",
                    "json_schema": { "name": "output", "schema": request.output_schema, "strict": strict }
                });
            }
            Strategy::ToolCall => {
                body["tools"] = json!([{
                    "type": "function",
                    "function": {
                        "name": "respond",
                        "description": "Return the result.",
                        "parameters": request.output_schema,
                        "strict": strict
                    }
                }]);
                body["tool_choice"] =
                    json!({ "type": "function", "function": { "name": "respond" } });
            }
            Strategy::JsonMode => body["response_format"] = json!({ "type": "json_object" }),
            Strategy::PromptOnly => {}
        }
        if let Some(t) = request.options.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(m) = request.options.max_tokens {
            body["max_tokens"] = json!(m);
        }
        body
    }

    async fn post(&self, body: &Value) -> Result<Value, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);
        let payload = serde_json::to_vec(body).expect("JSON values always serialize");
        let mut attempt = 0;
        loop {
            let mut builder = self
                .client
                .post(&url)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(payload.clone());
            if let Some(key) = &self.api_key {
                builder = builder.bearer_auth(key);
            }

            let (error, retry_after) = match builder.send().await {
                Ok(response) if response.status().is_success() => {
                    let bytes = response
                        .bytes()
                        .await
                        .map_err(|e| ProviderError::Transport(e.to_string()))?;
                    return serde_json::from_slice(&bytes)
                        .map_err(|e| ProviderError::InvalidResponse(e.to_string()));
                }
                Ok(response) => {
                    let code = response.status().as_u16();
                    let retry_after = response
                        .headers()
                        .get(reqwest::header::RETRY_AFTER)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.trim().parse::<u64>().ok())
                        .map(|s| Duration::from_secs(s).min(MAX_RETRY_AFTER));
                    let body = response.text().await.unwrap_or_default();
                    let error = ProviderError::Status { code, body };
                    if code != 429 && code < 500 {
                        return Err(error);
                    }
                    (error, retry_after)
                }
                Err(e) => (ProviderError::Transport(e.to_string()), None),
            };

            if attempt >= self.max_transport_retries {
                return Err(error);
            }
            let delay = retry_after.unwrap_or(self.initial_backoff * 2u32.pow(attempt));
            tokio::time::sleep(delay).await;
            attempt += 1;
        }
    }

    fn parse(&self, body: &Value, strategy: Strategy) -> Result<Response, ProviderError> {
        let choice = body
            .pointer("/choices/0")
            .ok_or_else(|| ProviderError::InvalidResponse("no choices in response".into()))?;
        let message = &choice["message"];

        let refusal = message["refusal"].as_str().filter(|r| !r.is_empty());
        let content = match (refusal, message.pointer("/tool_calls/0/function/arguments")) {
            (Some(refusal), _) => refusal.to_string(),
            (None, Some(Value::String(arguments))) => arguments.clone(),
            _ => {
                let text = message["content"].as_str().unwrap_or_default();
                match strategy {
                    Strategy::ToolCall => {
                        tool_call_in_text(text).unwrap_or_else(|| text.to_string())
                    }
                    _ => text.to_string(),
                }
            }
        };
        let finish = match (refusal, choice["finish_reason"].as_str()) {
            (Some(_), _) | (_, Some("content_filter")) => FinishReason::Refusal,
            (_, Some("length")) => FinishReason::Length,
            _ => FinishReason::Stop,
        };

        Ok(Response {
            content,
            model: body["model"].as_str().unwrap_or(&self.model).to_string(),
            finish,
            usage: Usage {
                input_tokens: body
                    .pointer("/usage/prompt_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                output_tokens: body
                    .pointer("/usage/completion_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            },
        })
    }
}

impl Provider for OpenAiCompatible {
    fn capabilities(&self) -> Capabilities {
        self.capabilities.clone()
    }

    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        let body = self.post(&self.body(&request)).await?;
        self.parse(&body, request.strategy)
    }
}

/// Arguments of a `respond` call that the model wrote as text instead of a
/// structured tool call — common with local models when the server's tool-call
/// parsing does not recognise the output.
fn tool_call_in_text(text: &str) -> Option<String> {
    let value = crate::json::extract_json(text).ok()?;
    if value.get("name")?.as_str()? != "respond" {
        return None;
    }
    match value.get("arguments")? {
        Value::String(arguments) => Some(arguments.clone()),
        arguments @ Value::Object(_) => Some(arguments.to_string()),
        _ => None,
    }
}
