//! Lenient extraction of a JSON value from model output.
//!
//! Accepts what models commonly wrap around JSON — Markdown code fences, prose
//! before and after, trailing commas — but never invents content: an unbalanced
//! (truncated) value is reported, not completed.

use serde_json::Value;

pub(crate) fn extract_json(text: &str) -> Result<Value, String> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str(trimmed) {
        return Ok(value);
    }

    let body = fenced_block(trimmed).unwrap_or(trimmed);
    let Some(start) = body.find(['{', '[']) else {
        return Err("no JSON object or array found".into());
    };
    let Some(len) = balanced_len(&body[start..]) else {
        return Err("JSON value is not closed (response truncated?)".into());
    };
    let candidate = &body[start..start + len];

    match serde_json::from_str(candidate) {
        Ok(value) => Ok(value),
        Err(first) => {
            serde_json::from_str(&strip_trailing_commas(candidate)).map_err(|_| first.to_string())
        }
    }
}

/// Content of the first Markdown code fence, if any.
fn fenced_block(text: &str) -> Option<&str> {
    let open = text.find("```")?;
    let after = &text[open + 3..];
    let content_start = after.find('\n')? + 1;
    let content = &after[content_start..];
    let close = content.find("```")?;
    Some(&content[..close])
}

/// Byte length of the leading object/array, honouring strings and escapes.
fn balanced_len(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in text.char_indices() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Removes commas directly before `}` or `]` outside of strings.
fn strip_trailing_commas(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
        } else if c == '"' {
            in_string = true;
        } else if c == ',' {
            let next = chars[i + 1..].iter().find(|c| !c.is_whitespace());
            if matches!(next, Some('}' | ']')) {
                continue;
            }
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn plain_json() {
        assert_eq!(extract_json(r#" {"a": 1} "#).unwrap(), json!({"a": 1}));
    }

    #[test]
    fn code_fence_and_prose() {
        let text = "Here you go:\n```json\n{\"a\": [1, 2]}\n```\nAnything else?";
        assert_eq!(extract_json(text).unwrap(), json!({"a": [1, 2]}));
    }

    #[test]
    fn prose_without_fence() {
        let text = r#"The answer is {"a": "x}y"} as requested."#;
        assert_eq!(extract_json(text).unwrap(), json!({"a": "x}y"}));
    }

    #[test]
    fn trailing_commas() {
        let text = r#"{"a": [1, 2,], "b": "c,]",}"#;
        assert_eq!(
            extract_json(text).unwrap(),
            json!({"a": [1, 2], "b": "c,]"})
        );
    }

    #[test]
    fn truncated_is_reported() {
        let err = extract_json(r#"{"a": "unfinished"#).unwrap_err();
        assert!(err.contains("not closed"));
    }

    #[test]
    fn no_json() {
        assert!(extract_json("I cannot help with that.").is_err());
    }
}
