//! Validation of a JSON value against the JSON Schema subset `schemars` generates.
//!
//! Unlike deserialization, which stops at the first problem, this collects every
//! violation with its path, so a repair request can name all of them at once.
//! Supported: `$ref` (local), `type`, `enum`, `const`, `properties`, `required`,
//! `additionalProperties`, `items`, `anyOf`, `oneOf`, `allOf`, numeric bounds,
//! string and array length. Other keywords (e.g. `format`) are ignored.

use std::fmt;

use serde_json::{Map, Value};

/// One way a value fails the schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    /// Location in the value, e.g. `$.items[2].price`.
    pub path: String,
    pub message: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

pub(crate) fn validate(schema: &Value, value: &Value) -> Vec<Violation> {
    let mut violations = Vec::new();
    Validator { root: schema }.check(schema, value, "$", &mut violations);
    violations
}

struct Validator<'a> {
    root: &'a Value,
}

impl Validator<'_> {
    fn check(&self, schema: &Value, value: &Value, path: &str, out: &mut Vec<Violation>) {
        let schema = match schema {
            Value::Bool(true) => return,
            Value::Bool(false) => return push(out, path, "no value is allowed here".into()),
            Value::Object(map) => map,
            _ => return,
        };

        if let Some(Value::String(reference)) = schema.get("$ref") {
            match reference
                .strip_prefix('#')
                .and_then(|p| self.root.pointer(p))
            {
                Some(target) => self.check(target, value, path, out),
                None => push(
                    out,
                    path,
                    format!("unresolvable schema reference {reference}"),
                ),
            }
        }

        if let Some(Value::Array(branches)) = schema.get("allOf") {
            for branch in branches {
                self.check(branch, value, path, out);
            }
        }
        for keyword in ["anyOf", "oneOf"] {
            if let Some(Value::Array(branches)) = schema.get(keyword) {
                self.check_branches(keyword, branches, value, path, out);
            }
        }

        if let Some(expected) = schema.get("const") {
            if value != expected {
                push(
                    out,
                    path,
                    format!("expected {expected}, got {}", short(value)),
                );
            }
        }
        if let Some(Value::Array(allowed)) = schema.get("enum") {
            if !allowed.contains(value) {
                let list = allowed
                    .iter()
                    .map(Value::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                push(
                    out,
                    path,
                    format!("expected one of {list}, got {}", short(value)),
                );
                return;
            }
        }

        if let Some(types) = schema.get("type") {
            if !type_matches(types, value) {
                push(
                    out,
                    path,
                    format!("expected {}, got {}", type_names(types), short(value)),
                );
                return;
            }
        }

        match value {
            Value::Object(object) => self.check_object(schema, object, path, out),
            Value::Array(items) => self.check_array(schema, items, path, out),
            Value::String(s) => check_string(schema, s, path, out),
            Value::Number(_) => check_number(schema, value, path, out),
            _ => {}
        }
    }

    fn check_branches(
        &self,
        keyword: &str,
        branches: &[Value],
        value: &Value,
        path: &str,
        out: &mut Vec<Violation>,
    ) {
        let results: Vec<Vec<Violation>> = branches
            .iter()
            .map(|branch| {
                let mut branch_out = Vec::new();
                self.check(branch, value, path, &mut branch_out);
                branch_out
            })
            .collect();
        let matching = results.iter().filter(|r| r.is_empty()).count();

        if matching == 0 {
            // A single near-miss is more useful than "matches no variant".
            if let Some(closest) = results.into_iter().min_by_key(Vec::len) {
                if branches.len() == 1 {
                    out.extend(closest);
                } else {
                    push(
                        out,
                        path,
                        format!("matches none of the allowed variants ({keyword})"),
                    );
                    out.extend(closest);
                }
            }
        } else if keyword == "oneOf" && matching > 1 {
            push(out, path, "matches more than one variant (oneOf)".into());
        }
    }

    fn check_object(
        &self,
        schema: &Map<String, Value>,
        object: &Map<String, Value>,
        path: &str,
        out: &mut Vec<Violation>,
    ) {
        let properties = schema.get("properties").and_then(Value::as_object);

        if let Some(Value::Array(required)) = schema.get("required") {
            for name in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(name) {
                    push(out, &child(path, name), "required field is missing".into());
                }
            }
        }

        for (name, field) in object {
            let field_path = child(path, name);
            match properties.and_then(|p| p.get(name)) {
                Some(field_schema) => self.check(field_schema, field, &field_path, out),
                None => match schema.get("additionalProperties") {
                    Some(Value::Bool(false)) => {
                        push(out, &field_path, "field is not allowed".into())
                    }
                    Some(extra @ Value::Object(_)) => self.check(extra, field, &field_path, out),
                    _ => {}
                },
            }
        }
    }

    fn check_array(
        &self,
        schema: &Map<String, Value>,
        items: &[Value],
        path: &str,
        out: &mut Vec<Violation>,
    ) {
        if let Some(min) = schema.get("minItems").and_then(Value::as_u64) {
            if (items.len() as u64) < min {
                push(
                    out,
                    path,
                    format!("expected at least {min} items, got {}", items.len()),
                );
            }
        }
        if let Some(max) = schema.get("maxItems").and_then(Value::as_u64) {
            if items.len() as u64 > max {
                push(
                    out,
                    path,
                    format!("expected at most {max} items, got {}", items.len()),
                );
            }
        }
        if let Some(item_schema) = schema.get("items") {
            for (i, item) in items.iter().enumerate() {
                self.check(item_schema, item, &format!("{path}[{i}]"), out);
            }
        }
    }
}

fn check_string(schema: &Map<String, Value>, s: &str, path: &str, out: &mut Vec<Violation>) {
    let len = s.chars().count() as u64;
    if let Some(min) = schema.get("minLength").and_then(Value::as_u64) {
        if len < min {
            push(
                out,
                path,
                format!("expected at least {min} characters, got {len}"),
            );
        }
    }
    if let Some(max) = schema.get("maxLength").and_then(Value::as_u64) {
        if len > max {
            push(
                out,
                path,
                format!("expected at most {max} characters, got {len}"),
            );
        }
    }
}

fn check_number(schema: &Map<String, Value>, value: &Value, path: &str, out: &mut Vec<Violation>) {
    let Some(n) = value.as_f64() else { return };
    let bound = |key| schema.get(key).and_then(Value::as_f64);
    if let Some(min) = bound("minimum").filter(|&min| n < min) {
        push(out, path, format!("expected a value >= {min}, got {value}"));
    }
    if let Some(max) = bound("maximum").filter(|&max| n > max) {
        push(out, path, format!("expected a value <= {max}, got {value}"));
    }
    if let Some(min) = bound("exclusiveMinimum").filter(|&min| n <= min) {
        push(out, path, format!("expected a value > {min}, got {value}"));
    }
    if let Some(max) = bound("exclusiveMaximum").filter(|&max| n >= max) {
        push(out, path, format!("expected a value < {max}, got {value}"));
    }
}

fn type_matches(types: &Value, value: &Value) -> bool {
    match types {
        Value::String(t) => single_type_matches(t, value),
        Value::Array(ts) => ts
            .iter()
            .filter_map(Value::as_str)
            .any(|t| single_type_matches(t, value)),
        _ => true,
    }
}

fn single_type_matches(t: &str, value: &Value) -> bool {
    match t {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "number" => value.is_number(),
        "integer" => {
            value.is_i64() || value.is_u64() || value.as_f64().is_some_and(|n| n.fract() == 0.0)
        }
        _ => true,
    }
}

fn type_names(types: &Value) -> String {
    match types {
        Value::Array(ts) => ts
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" or "),
        Value::String(t) => t.clone(),
        other => other.to_string(),
    }
}

fn child(path: &str, name: &str) -> String {
    format!("{path}.{name}")
}

fn push(out: &mut Vec<Violation>, path: &str, message: String) {
    out.push(Violation {
        path: path.into(),
        message,
    });
}

/// Value rendering for messages, cut so a huge value does not flood the repair prompt.
fn short(value: &Value) -> String {
    const MAX: usize = 80;
    let text = value.to_string();
    if text.chars().count() <= MAX {
        text
    } else {
        let cut: String = text.chars().take(MAX).collect();
        format!("{cut}…")
    }
}
