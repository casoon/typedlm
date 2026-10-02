//! Compiled programs: a program's configuration as a reviewable JSON artefact.
//!
//! Instructions, worked examples and generation settings that were found to work — by
//! hand today, by an optimiser later — are stored in one file next to the code. Production
//! loads the file and runs it; nothing that produced it needs to ship.
//!
//! ```ignore
//! // After tuning:
//! program.compile().with_provenance(&report).save("classify.typedlm.json")?;
//!
//! // In production, embedded and checked when the binary is built:
//! const CLASSIFY: &str = include_str!("../classify.typedlm.json");
//! let classify = CompiledProgram::from_json(CLASSIFY)?.program::<ClassifyTicket, _>(provider)?;
//! ```

use std::fmt;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::schema::validate;
use crate::{
    Demonstration, GenerationOptions, Program, Provider, Signature, Strategy, output_schema,
};

/// Version of the file format, not of the crate.
pub const FORMAT: u32 = 1;

/// A program's configuration, independent of the provider it runs on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledProgram {
    pub format: u32,
    /// `Signature::NAME`.
    pub program: String,
    /// SHA-256 over the name and the output schema; a changed output type no longer
    /// matches and the artefact is refused.
    pub signature: String,
    /// Replaces the instructions generated from the description; `None` keeps them.
    pub instructions: Option<String>,
    pub demonstrations: Vec<Demonstration>,
    /// Forced strategy; `None` uses the strongest the provider supports.
    pub strategy: Option<Strategy>,
    pub generation: GenerationOptions,
    pub max_repairs: usize,
    /// The evaluation this configuration was accepted on, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

/// Where a compiled program's quality claim comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Requested model of the run, or the model that answered.
    pub model: Option<String>,
    pub metric: String,
    pub score: f64,
    pub interval: (f64, f64),
    pub examples: usize,
    /// SHA-256 of the dataset the run used.
    pub dataset: String,
}

#[cfg(feature = "eval")]
impl From<&crate::eval::Report> for Provenance {
    fn from(report: &crate::eval::Report) -> Self {
        Self {
            model: report.label.clone().or_else(|| report.model.clone()),
            metric: report.metric.clone(),
            score: report.score,
            interval: report.interval,
            examples: report.examples,
            dataset: report.dataset.clone(),
        }
    }
}

/// Why a compiled program cannot be used.
#[derive(Debug)]
#[non_exhaustive]
pub enum CompileError {
    Io {
        path: String,
        message: String,
    },
    /// Not a compiled program, or a newer format.
    Invalid {
        message: String,
    },
    /// Built for another signature, or the output type changed since.
    SignatureMismatch {
        program: String,
        expected: String,
        found: String,
    },
    /// A worked example does not fit the input or output type.
    InvalidDemonstration {
        index: usize,
        message: String,
    },
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "{path}: {message}"),
            Self::Invalid { message } => write!(f, "not a compiled program: {message}"),
            Self::SignatureMismatch {
                program,
                expected,
                found,
            } => write!(
                f,
                "compiled for {found}, not for {program} ({expected}); the output type or \
                 the name changed — compile the program again"
            ),
            Self::InvalidDemonstration { index, message } => {
                write!(f, "demonstration {index}: {message}")
            }
        }
    }
}

impl std::error::Error for CompileError {}

/// Fingerprint of a signature: SHA-256 over its name and output schema.
pub fn signature_hash<S: Signature>() -> String {
    let fingerprint = serde_json::json!({ "name": S::NAME, "output": output_schema::<S>() });
    let bytes = serde_json::to_vec(&fingerprint).expect("JSON values always serialize");
    crate::sha256::hex(&crate::sha256::sha256(&bytes))
}

impl CompiledProgram {
    /// Attaches the evaluation that accepted this configuration.
    pub fn with_provenance(mut self, provenance: impl Into<Provenance>) -> Self {
        self.provenance = Some(provenance.into());
        self
    }

    /// Pretty JSON with a trailing newline, so diffs stay readable.
    pub fn to_json(&self) -> String {
        let mut json = serde_json::to_string_pretty(self).expect("compiled programs serialize");
        json.push('\n');
        json
    }

    /// Parses a compiled program without checking it against a signature; that happens in
    /// [`program`](CompiledProgram::program).
    pub fn from_json(json: &str) -> Result<Self, CompileError> {
        let compiled: Self = serde_json::from_str(json).map_err(|e| CompileError::Invalid {
            message: e.to_string(),
        })?;
        if compiled.format > FORMAT {
            return Err(CompileError::Invalid {
                message: format!(
                    "format {} is newer than this version of typedlm reads ({FORMAT})",
                    compiled.format
                ),
            });
        }
        Ok(compiled)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, CompileError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| io(path, e))?;
        Self::from_json(&text)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), CompileError> {
        let path = path.as_ref();
        std::fs::write(path, self.to_json()).map_err(|e| io(path, e))
    }

    /// Checks the artefact against `S` and builds the program on `provider`: the signature
    /// must match, and every demonstration must fit the input and output types.
    pub fn program<S, P>(&self, provider: P) -> Result<Program<S, P>, CompileError>
    where
        S: Signature + DeserializeOwned,
        P: Provider,
    {
        self.check::<S>()?;
        Ok(Program::from_parts(provider, self))
    }

    fn check<S: Signature + DeserializeOwned>(&self) -> Result<(), CompileError> {
        let expected = signature_hash::<S>();
        if self.program != S::NAME || self.signature != expected {
            return Err(CompileError::SignatureMismatch {
                program: S::NAME.into(),
                expected: short(&expected),
                found: format!("{} ({})", self.program, short(&self.signature)),
            });
        }
        let schema = output_schema::<S>();
        for (index, demo) in self.demonstrations.iter().enumerate() {
            let invalid = |message: String| CompileError::InvalidDemonstration { index, message };
            serde_json::from_value::<S>(demo.input.clone())
                .map_err(|e| invalid(format!("input does not fit {}: {e}", S::NAME)))?;
            let violations = validate(&schema, &demo.output);
            if let Some(first) = violations.first() {
                return Err(invalid(format!("output violates the schema: {first}")));
            }
            serde_json::from_value::<S::Output>(demo.output.clone())
                .map_err(|e| invalid(format!("output does not deserialize: {e}")))?;
        }
        Ok(())
    }
}

fn short(hash: &str) -> String {
    hash.chars().take(12).collect()
}

fn io(path: &Path, error: std::io::Error) -> CompileError {
    CompileError::Io {
        path: path.display().to_string(),
        message: error.to_string(),
    }
}
