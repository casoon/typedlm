//! Providers for tests that run without a model: fixed answers, closures, answers
//! generated from the schema, and recording real interactions to replay them offline.
//!
//! ```ignore
//! use typedlm::testing::TestProvider;
//!
//! let provider = TestProvider::answers([r#"{"urgency": "High"}"#]);
//! let out = Program::<ClassifyTicket, _>::new(&provider).run("db down").await?;
//! assert_eq!(provider.requests()[0].input["text"], "db down");
//! ```

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    Capabilities, FinishReason, Provider, ProviderError, Request, Response, SchemaDialect,
    Strategy, Usage,
};

fn generic_capabilities() -> Capabilities {
    Capabilities::new(vec![Strategy::NativeSchema], SchemaDialect::Generic)
}

fn answer(content: String) -> Response {
    Response {
        content,
        model: "test".into(),
        finish: FinishReason::Stop,
        usage: Usage::default(),
    }
}

/// Answers with fixed texts in order, or with a schema-valid value, and keeps every
/// request for assertions.
pub struct TestProvider {
    source: Source,
    capabilities: Capabilities,
    requests: Mutex<Vec<Request>>,
}

enum Source {
    Answers(Mutex<VecDeque<String>>),
    Valid,
}

impl TestProvider {
    /// Returns the answers in order; one more request than answers is an error.
    pub fn answers<I, A>(answers: I) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<String>,
    {
        let answers = answers.into_iter().map(Into::into).collect();
        Self::with_source(Source::Answers(Mutex::new(answers)))
    }

    /// Answers every request with a value generated from its output schema: required
    /// fields only, the first enum variant, the smallest allowed number. Useful to run
    /// the code around a program without caring what the model would say.
    pub fn valid() -> Self {
        Self::with_source(Source::Valid)
    }

    fn with_source(source: Source) -> Self {
        Self {
            source,
            capabilities: generic_capabilities(),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Default: `NativeSchema` with the generic dialect.
    pub fn capabilities(mut self, capabilities: Capabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Every request received so far, oldest first.
    pub fn requests(&self) -> Vec<Request> {
        self.requests.lock().expect("lock").clone()
    }
}

impl Provider for TestProvider {
    fn capabilities(&self) -> Capabilities {
        self.capabilities.clone()
    }

    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        let content = match &self.source {
            Source::Answers(answers) => {
                answers.lock().expect("lock").pop_front().ok_or_else(|| {
                    ProviderError::InvalidResponse("TestProvider has no answers left".into())
                })?
            }
            Source::Valid => sample(&request.output_schema, &request.output_schema, 0).to_string(),
        };
        self.requests.lock().expect("lock").push(request);
        Ok(answer(content))
    }
}

/// Answers through a closure that sees the whole request, e.g. to answer by input.
pub struct FnProvider<F> {
    answer: F,
    capabilities: Capabilities,
}

impl<F> FnProvider<F>
where
    F: Fn(&Request) -> Result<String, ProviderError> + Send + Sync,
{
    pub fn new(answer: F) -> Self {
        Self {
            answer,
            capabilities: generic_capabilities(),
        }
    }

    /// Default: `NativeSchema` with the generic dialect.
    pub fn capabilities(mut self, capabilities: Capabilities) -> Self {
        self.capabilities = capabilities;
        self
    }
}

impl<F> Provider for FnProvider<F>
where
    F: Fn(&Request) -> Result<String, ProviderError> + Send + Sync,
{
    fn capabilities(&self) -> Capabilities {
        self.capabilities.clone()
    }

    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        (self.answer)(&request).map(answer)
    }
}

/// A value that satisfies `schema` (the subset `schemars` generates).
fn sample(schema: &Value, root: &Value, depth: usize) -> Value {
    let Value::Object(map) = schema else {
        return Value::Null;
    };
    if let Some(Value::String(reference)) = map.get("$ref") {
        let target = reference.strip_prefix('#').and_then(|p| root.pointer(p));
        return match target {
            Some(target) if depth < 32 => sample(target, root, depth + 1),
            _ => Value::Null,
        };
    }
    if let Some(value) = map.get("const") {
        return value.clone();
    }
    if let Some(Value::Array(values)) = map.get("enum") {
        return values.first().cloned().unwrap_or(Value::Null);
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(branches)) = map.get(key) {
            // Prefer a non-null branch, but stop recursing once nesting gets deep.
            let not_null = |b: &&Value| b.get("type").and_then(Value::as_str) != Some("null");
            let pick = if depth < 8 {
                branches.iter().find(not_null).or(branches.first())
            } else {
                branches.iter().find(|b| !not_null(b)).or(branches.first())
            };
            return pick.map_or(Value::Null, |b| sample(b, root, depth + 1));
        }
    }

    let kind = match map.get("type") {
        Some(Value::String(t)) => t.as_str(),
        Some(Value::Array(types)) => types
            .iter()
            .filter_map(Value::as_str)
            .find(|t| *t != "null")
            .unwrap_or("null"),
        _ if map.contains_key("properties") => "object",
        _ => "null",
    };
    let number = |integer: bool| {
        let get = |k| map.get(k).and_then(Value::as_f64);
        let mut n = get("minimum").unwrap_or(0.0);
        if let Some(min) = get("exclusiveMinimum") {
            n = n.max(if integer {
                min.floor() + 1.0
            } else {
                min + 1.0
            });
        }
        if let Some(max) = get("maximum").or(get("exclusiveMaximum")) {
            n = n.min(max);
        }
        n
    };
    match kind {
        "object" => {
            let mut object = Map::new();
            let required: Vec<&str> = map
                .get("required")
                .and_then(Value::as_array)
                .map(|r| r.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            if let Some(Value::Object(properties)) = map.get("properties") {
                for name in required {
                    if let Some(property) = properties.get(name) {
                        object.insert(name.into(), sample(property, root, depth + 1));
                    }
                }
            }
            Value::Object(object)
        }
        "array" => {
            let count = map.get("minItems").and_then(Value::as_u64).unwrap_or(0);
            let item = map.get("items").cloned().unwrap_or(Value::Null);
            Value::Array((0..count).map(|_| sample(&item, root, depth + 1)).collect())
        }
        "string" => {
            let min = map.get("minLength").and_then(Value::as_u64).unwrap_or(0) as usize;
            let max = map
                .get("maxLength")
                .and_then(Value::as_u64)
                .map(|m| m as usize);
            let mut text = "text".repeat(min.div_ceil(4).max(1));
            text.truncate(max.unwrap_or(usize::MAX).max(min));
            Value::String(text)
        }
        "integer" => Value::from(number(true) as i64),
        "number" => serde_json::Number::from_f64(number(false)).map_or(Value::Null, Value::Number),
        "boolean" => Value::Bool(false),
        _ => Value::Null,
    }
}

const CASSETTE_VERSION: u32 = 1;

/// Recorded interactions, as stored on disk. Requests are keyed by a SHA-256 of their
/// JSON, so any change to instructions, input, schema or options misses the recording.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Cassette {
    version: u32,
    capabilities: Capabilities,
    interactions: Vec<Interaction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Interaction {
    key: String,
    request: Request,
    response: Response,
}

fn key(request: &Request) -> String {
    let json = serde_json::to_vec(request).expect("requests serialize to JSON");
    crate::sha256::hex(&crate::sha256::sha256(&json))
}

fn recording_error(path: &Path, message: impl std::fmt::Display) -> ProviderError {
    ProviderError::Recording(format!("{}: {message}", path.display()))
}

fn read_cassette(path: &Path) -> Result<Cassette, ProviderError> {
    let text = std::fs::read_to_string(path).map_err(|e| recording_error(path, e))?;
    let cassette: Cassette = serde_json::from_str(&text).map_err(|e| recording_error(path, e))?;
    if cassette.version != CASSETTE_VERSION {
        return Err(recording_error(
            path,
            format!("unsupported version {}", cassette.version),
        ));
    }
    Ok(cassette)
}

/// Passes requests to a real provider and writes each request with its response to a
/// JSON file, for [`Replay`]. Nothing outside the request is stored: no API keys, no
/// headers. Requests already in the file are overwritten with the new answer.
///
/// The file holds inputs and model answers; treat it like test data.
pub struct Recorder<P> {
    inner: P,
    path: PathBuf,
    interactions: Mutex<BTreeMap<String, Interaction>>,
}

impl<P: Provider> Recorder<P> {
    /// Keeps interactions already recorded at `path`.
    pub fn new(path: impl Into<PathBuf>, inner: P) -> Result<Self, ProviderError> {
        let path = path.into();
        let interactions = if path.exists() {
            read_cassette(&path)?
                .interactions
                .into_iter()
                .map(|i| (i.key.clone(), i))
                .collect()
        } else {
            BTreeMap::new()
        };
        Ok(Self {
            inner,
            path,
            interactions: Mutex::new(interactions),
        })
    }

    fn save(&self, interactions: &BTreeMap<String, Interaction>) -> Result<(), ProviderError> {
        let cassette = Cassette {
            version: CASSETTE_VERSION,
            capabilities: self.inner.capabilities(),
            interactions: interactions.values().cloned().collect(),
        };
        if let Some(parent) = self.path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| recording_error(&self.path, e))?;
        }
        let mut json = serde_json::to_string_pretty(&cassette).expect("cassettes serialize");
        json.push('\n');
        std::fs::write(&self.path, json).map_err(|e| recording_error(&self.path, e))
    }
}

impl<P: Provider> Provider for Recorder<P> {
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        let response = self.inner.complete(request.clone()).await?;
        let mut interactions = self.interactions.lock().expect("lock");
        let key = key(&request);
        interactions.insert(
            key.clone(),
            Interaction {
                key,
                request,
                response: response.clone(),
            },
        );
        self.save(&interactions)?;
        Ok(response)
    }
}

/// Answers from a file written by [`Recorder`], offline and deterministic. It reports
/// the recorded provider's capabilities, so programs build the same requests. A request
/// that was never recorded is an error naming the file.
pub struct Replay {
    path: PathBuf,
    capabilities: Capabilities,
    responses: BTreeMap<String, Response>,
}

impl Replay {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, ProviderError> {
        let path = path.into();
        let cassette = read_cassette(&path)?;
        Ok(Self {
            path,
            capabilities: cassette.capabilities,
            responses: cassette
                .interactions
                .into_iter()
                .map(|i| (i.key, i.response))
                .collect(),
        })
    }

    /// Number of recorded interactions.
    pub fn len(&self) -> usize {
        self.responses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.responses.is_empty()
    }
}

impl Provider for Replay {
    fn capabilities(&self) -> Capabilities {
        self.capabilities.clone()
    }

    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        self.responses.get(&key(&request)).cloned().ok_or_else(|| {
            recording_error(
                &self.path,
                "no recording for this request; instructions, input, schema or options \
                 changed since recording — record again with Recorder",
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn samples_satisfy_their_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "urgency": {"type": "string", "enum": ["Low", "High"]},
                "note": {"type": ["string", "null"]},
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "ratio": {"type": "number", "exclusiveMinimum": 0.5},
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                "code": {"type": "string", "minLength": 6, "maxLength": 6},
                "item": {"anyOf": [{"$ref": "#/$defs/Item"}, {"type": "null"}]},
                "kind": {"const": "fixed"}
            },
            "required": ["urgency", "note", "score", "ratio", "tags", "code", "item", "kind"],
            "$defs": {"Item": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}}
        });
        let value = sample(&schema, &schema, 0);
        assert_eq!(crate::schema::validate(&schema, &value), vec![]);
        assert_eq!(value["item"], json!({"ok": false}));
        assert_eq!(value["code"].as_str().unwrap().len(), 6);
    }

    #[test]
    fn recursive_schemas_terminate() {
        let schema = json!({
            "$ref": "#/$defs/Node",
            "$defs": {"Node": {"type": "object", "properties": {
                "next": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]}
            }, "required": ["next"]}}
        });
        let value = sample(&schema, &schema, 0);
        assert_eq!(crate::schema::validate(&schema, &value), vec![]);
    }
}
