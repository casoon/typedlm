use std::fmt;

use crate::ProviderError;
use crate::schema::Violation;

/// Why a program run produced no typed output.
///
/// Variants carrying `response` hold raw model output, which may contain personal
/// data. `Display` prints no model output — neither the response nor values quoted
/// in violation or deserialization messages; read the fields for those.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Provider(ProviderError),
    /// The input could not be serialized to JSON.
    InvalidInput {
        message: String,
    },
    /// The model declined to answer. Not repairable.
    Refusal {
        model: String,
        message: String,
    },
    /// The output token limit cut the answer off. Not repairable by asking again
    /// with the same limit.
    Truncated {
        model: String,
    },
    /// No JSON value could be extracted from the answer.
    InvalidJson {
        reason: String,
        response: String,
    },
    /// The JSON does not satisfy the output schema.
    SchemaViolation {
        violations: Vec<Violation>,
        response: String,
    },
    /// The JSON satisfies the schema but does not deserialize into the output type.
    Deserialize {
        message: String,
        response: String,
    },
    /// The output failed the signature's own validation.
    Validation {
        errors: Vec<String>,
    },
    /// Every attempt failed; holds each attempt's error, the last one last.
    RetryLimitExceeded {
        attempts: Vec<Error>,
    },
}

impl Error {
    /// Stable lowercase name of the variant, for traces and metrics.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Provider(_) => "provider",
            Self::InvalidInput { .. } => "invalid_input",
            Self::Refusal { .. } => "refusal",
            Self::Truncated { .. } => "truncated",
            Self::InvalidJson { .. } => "invalid_json",
            Self::SchemaViolation { .. } => "schema_violation",
            Self::Deserialize { .. } => "deserialize",
            Self::Validation { .. } => "validation",
            Self::RetryLimitExceeded { .. } => "retry_limit_exceeded",
        }
    }

    /// Whether telling the model what was wrong can fix the answer.
    pub fn is_repairable(&self) -> bool {
        matches!(
            self,
            Self::InvalidJson { .. }
                | Self::SchemaViolation { .. }
                | Self::Deserialize { .. }
                | Self::Validation { .. }
        )
    }

    /// Feedback for the model describing what to correct, for repairable errors.
    pub fn repair_feedback(&self) -> Option<String> {
        let details = match self {
            Self::InvalidJson { reason, .. } => {
                format!("It is not valid JSON ({reason}). Respond with a single JSON value only.")
            }
            Self::SchemaViolation { violations, .. } => {
                let lines: Vec<String> = violations.iter().map(|v| format!("- {v}")).collect();
                format!("It violates the output schema:\n{}", lines.join("\n"))
            }
            Self::Deserialize { message, .. } => {
                format!("It does not match the expected structure: {message}")
            }
            Self::Validation { errors } => {
                let lines: Vec<String> = errors.iter().map(|e| format!("- {e}")).collect();
                format!("It fails validation:\n{}", lines.join("\n"))
            }
            _ => return None,
        };
        Some(format!(
            "Your previous response does not satisfy the output contract. {details}\n\
             Return a corrected response."
        ))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(e) => write!(f, "{e}"),
            Self::InvalidInput { message } => write!(f, "input cannot be serialized: {message}"),
            Self::Refusal { model, .. } => write!(f, "model {model} refused to answer"),
            Self::Truncated { model } => write!(f, "answer of model {model} was truncated"),
            Self::InvalidJson { reason, .. } => write!(f, "answer is not valid JSON: {reason}"),
            Self::SchemaViolation { violations, .. } => {
                let paths: Vec<&str> = violations.iter().map(|v| v.path.as_str()).collect();
                write!(
                    f,
                    "answer violates the output schema at {}",
                    paths.join(", ")
                )
            }
            Self::Deserialize { .. } => {
                write!(f, "answer does not deserialize into the output type")
            }
            Self::Validation { errors } => {
                write!(f, "answer failed validation: {}", errors.join("; "))
            }
            Self::RetryLimitExceeded { attempts } => {
                write!(f, "no valid answer after {} attempts", attempts.len())?;
                if let Some(last) = attempts.last() {
                    write!(f, ", last error: {last}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(e) => Some(e),
            Self::RetryLimitExceeded { attempts } => attempts
                .last()
                .map(|e| e as &(dyn std::error::Error + 'static)),
            _ => None,
        }
    }
}

impl From<ProviderError> for Error {
    fn from(e: ProviderError) -> Self {
        Self::Provider(e)
    }
}
