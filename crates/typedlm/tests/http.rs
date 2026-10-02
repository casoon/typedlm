#![cfg(feature = "http")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use typedlm::http::OpenAiCompatible;
use typedlm::prelude::*;
use typedlm::{Capabilities, ProviderError, SchemaDialect, Strategy};

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Invoice {
    number: String,
    total_cents: u64,
    note: Option<String>,
}

/// Extract the invoice.
#[derive(TypedLm, Serialize)]
#[lm(output = Invoice)]
struct ExtractInvoice {
    text: String,
}

struct Reply {
    status: u16,
    headers: &'static str,
    body: String,
}

fn ok(body: Value) -> Reply {
    Reply {
        status: 200,
        headers: "",
        body: body.to_string(),
    }
}

fn completion(message: Value, finish: &str) -> Reply {
    ok(json!({
        "model": "gpt-test-2026",
        "choices": [{ "index": 0, "message": message, "finish_reason": finish }],
        "usage": { "prompt_tokens": 50, "completion_tokens": 7 }
    }))
}

/// Serves the replies in order and records each request (headers lowercased, body JSON).
async fn serve(replies: Vec<Reply>) -> (String, Arc<Mutex<Vec<(String, Value)>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    tokio::spawn(async move {
        for reply in replies {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            let (head, body) = loop {
                let mut chunk = [0u8; 4096];
                let n = socket.read(&mut chunk).await.unwrap();
                buf.extend_from_slice(&chunk[..n]);
                let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
                    continue;
                };
                let head = String::from_utf8_lossy(&buf[..end]).to_lowercase();
                let len: usize = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .map(|v| v.trim().parse().unwrap())
                    .unwrap_or(0);
                if buf.len() >= end + 4 + len {
                    break (head, buf[end + 4..end + 4 + len].to_vec());
                }
            };
            log.lock()
                .unwrap()
                .push((head, serde_json::from_slice(&body).unwrap()));
            let response = format!(
                "HTTP/1.1 {} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n{}\r\n{}",
                reply.status,
                reply.body.len(),
                reply.headers,
                reply.body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        }
    });
    (url, seen)
}

#[tokio::test]
async fn native_schema_round_trip() {
    let answer = json!({"role": "assistant", "content": "{\"number\": \"R-17\", \"total_cents\": 11900, \"note\": null}"});
    let (url, seen) = serve(vec![completion(answer, "stop")]).await;
    let provider = OpenAiCompatible::new(url, "gpt-test").api_key("sk-test");

    let execution = Program::<ExtractInvoice, _>::new(provider)
        .temperature(0.0)
        .execute("Invoice R-17, total 119.00 EUR")
        .await
        .unwrap();
    assert_eq!(execution.output.total_cents, 11900);
    assert_eq!(execution.model, "gpt-test-2026");
    assert_eq!(execution.usage.output_tokens, 7);

    let (head, body) = seen.lock().unwrap()[0].clone();
    assert!(head.starts_with("post /v1/chat/completions"));
    assert!(head.contains("authorization: bearer sk-test"));
    assert_eq!(body["model"], "gpt-test");
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(
        body["messages"][1]["content"],
        r#"{"text":"Invoice R-17, total 119.00 EUR"}"#
    );
    let format = &body["response_format"];
    assert_eq!(format["type"], "json_schema");
    assert_eq!(format["json_schema"]["strict"], true);
    assert_eq!(
        format["json_schema"]["schema"]["required"],
        json!(["note", "number", "total_cents"])
    );
}

#[tokio::test]
async fn tool_call_strategy_reads_arguments() {
    let answer = json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{ "id": "c1", "type": "function", "function": {
            "name": "respond", "arguments": "{\"number\": \"A\", \"total_cents\": 1, \"note\": \"x\"}"
        }}]
    });
    let (url, seen) = serve(vec![completion(answer, "tool_calls")]).await;
    let program = Program::<ExtractInvoice, _>::new(OpenAiCompatible::new(url, "m"))
        .strategy(Strategy::ToolCall);

    let out = program.run("x").await.unwrap();
    assert_eq!(out.note.as_deref(), Some("x"));
    let body = seen.lock().unwrap()[0].1.clone();
    assert_eq!(body["tool_choice"]["function"]["name"], "respond");
    assert!(body.get("response_format").is_none());
}

/// What to do with an invoice.
#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
enum InvoiceAction {
    /// Pay the invoice now.
    Pay { number: String },
    /// Ask a human to review it.
    Review,
}

/// Decide what to do with an invoice.
#[derive(TypedLm, Serialize)]
#[lm(output = InvoiceAction)]
struct DecideInvoice {
    text: String,
}

#[tokio::test]
async fn action_enums_become_native_tools() {
    let pay = json!({"role": "assistant", "content": null, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "Pay", "arguments": "{\"number\": \"R-1\"}"}}
    ]});
    let review = json!({"role": "assistant", "content": null, "tool_calls": [
        {"id": "2", "type": "function", "function": {"name": "Review", "arguments": "{}"}}
    ]});
    let as_text = json!({"role": "assistant", "content": "{\"name\": \"Pay\", \"arguments\": {\"number\": \"R-2\"}}"});
    let (url, seen) = serve(vec![
        completion(pay, "tool_calls"),
        completion(review, "tool_calls"),
        completion(as_text, "stop"),
    ])
    .await;
    let program = Program::<DecideInvoice, _>::new(OpenAiCompatible::new(url, "m"))
        .strategy(Strategy::ToolCall);

    assert_eq!(
        program.run("x").await.unwrap(),
        InvoiceAction::Pay {
            number: "R-1".into()
        }
    );
    assert_eq!(program.run("x").await.unwrap(), InvoiceAction::Review);
    assert_eq!(
        program.run("x").await.unwrap(),
        InvoiceAction::Pay {
            number: "R-2".into()
        }
    );

    let body = seen.lock().unwrap()[0].1.clone();
    assert_eq!(body["tool_choice"], "required");
    let tools = body["tools"].as_array().unwrap();
    let names: Vec<&str> = tools
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["Pay", "Review"]);
    assert_eq!(tools[0]["function"]["description"], "Pay the invoice now.");
    assert_eq!(
        tools[0]["function"]["parameters"]["additionalProperties"], false,
        "strict dialect"
    );
    assert_eq!(tools[0]["function"]["strict"], true);
}

#[tokio::test]
async fn demonstrations_precede_the_input() {
    let answer = json!({"role": "assistant", "content": "{\"number\": \"B\", \"total_cents\": 2, \"note\": null}"});
    let (url, seen) = serve(vec![completion(answer, "stop")]).await;
    let example = Invoice {
        number: "A".into(),
        total_cents: 1,
        note: None,
    };
    Program::<ExtractInvoice, _>::new(OpenAiCompatible::new(url, "m"))
        .demonstration("Invoice A, 0.01 EUR", &example)
        .run("Invoice B, 0.02 EUR")
        .await
        .unwrap();

    let body = seen.lock().unwrap()[0].1.clone();
    let roles: Vec<&str> = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["system", "user", "assistant", "user"]);
    assert_eq!(
        body["messages"][1]["content"],
        r#"{"text":"Invoice A, 0.01 EUR"}"#
    );
    assert_eq!(
        body["messages"][2]["content"],
        r#"{"note":null,"number":"A","total_cents":1}"#
    );
    assert_eq!(
        body["messages"][3]["content"],
        r#"{"text":"Invoice B, 0.02 EUR"}"#
    );
}

#[tokio::test]
async fn tool_call_written_as_text_is_unwrapped() {
    let text =
        r#"{"name": "respond", "arguments": {"number": "A", "total_cents": 2, "note": null}}"#;
    let answer = json!({"role": "assistant", "content": text});
    let (url, _) = serve(vec![completion(answer, "stop")]).await;
    let execution = Program::<ExtractInvoice, _>::new(OpenAiCompatible::new(url, "m"))
        .strategy(Strategy::ToolCall)
        .execute("x")
        .await
        .unwrap();
    assert_eq!(execution.output.total_cents, 2);
    assert_eq!(execution.attempts, 1);
}

#[tokio::test]
async fn repair_turns_become_messages() {
    let bad = json!({"role": "assistant", "content": "{\"number\": \"A\"}"});
    let good = json!({"role": "assistant", "content": "{\"number\": \"A\", \"total_cents\": 5, \"note\": null}"});
    let (url, seen) = serve(vec![completion(bad, "stop"), completion(good, "stop")]).await;

    let execution = Program::<ExtractInvoice, _>::new(OpenAiCompatible::new(url, "m"))
        .execute("x")
        .await
        .unwrap();
    assert_eq!(execution.attempts, 2);

    let second = seen.lock().unwrap()[1].1.clone();
    let messages = second["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(messages[2]["content"], "{\"number\": \"A\"}");
    assert!(
        messages[3]["content"]
            .as_str()
            .unwrap()
            .contains("$.total_cents: required field is missing")
    );
}

#[tokio::test]
async fn refusal_field_and_length_map_to_finish_reasons() {
    let refusal = json!({"role": "assistant", "content": null, "refusal": "I can't do that."});
    let truncated = json!({"role": "assistant", "content": "{\"number\": \"A"});
    let (url, _) = serve(vec![
        completion(refusal, "stop"),
        completion(truncated, "length"),
    ])
    .await;
    let program = Program::<ExtractInvoice, _>::new(OpenAiCompatible::new(url, "m"));

    let err = program.run("x").await.unwrap_err();
    assert!(matches!(err, Error::Refusal { ref message, .. } if message == "I can't do that."));
    assert!(matches!(
        program.run("x").await.unwrap_err(),
        Error::Truncated { .. }
    ));
}

#[tokio::test]
async fn transport_retries_on_429_and_5xx() {
    let throttled = Reply {
        status: 429,
        headers: "retry-after: 0\r\n",
        body: "{}".into(),
    };
    let unavailable = Reply {
        status: 503,
        headers: "",
        body: "{}".into(),
    };
    let answer = json!({"role": "assistant", "content": "{\"number\": \"A\", \"total_cents\": 1, \"note\": null}"});
    let (url, seen) = serve(vec![throttled, unavailable, completion(answer, "stop")]).await;
    let provider = OpenAiCompatible::new(url, "m").initial_backoff(Duration::from_millis(1));

    let execution = Program::<ExtractInvoice, _>::new(provider)
        .execute("x")
        .await
        .unwrap();
    assert_eq!(
        execution.attempts, 1,
        "transport retries are not repair attempts"
    );
    assert_eq!(seen.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn client_errors_are_not_retried() {
    let bad_request = Reply {
        status: 400,
        headers: "",
        body: r#"{"error":"bad schema"}"#.into(),
    };
    let (url, seen) = serve(vec![bad_request]).await;
    let err = Program::<ExtractInvoice, _>::new(OpenAiCompatible::new(url, "m"))
        .run("x")
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Provider(ProviderError::Status { code: 400, ref body }) if body.contains("bad schema")
    ));
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn generic_dialect_sends_non_strict_schema() {
    let answer = json!({"role": "assistant", "content": "{\"number\": \"A\", \"total_cents\": 1}"});
    let (url, seen) = serve(vec![completion(answer, "stop")]).await;
    let provider = OpenAiCompatible::new(url, "llama").capabilities(Capabilities::new(
        vec![Strategy::NativeSchema],
        SchemaDialect::Generic,
    ));
    let out = Program::<ExtractInvoice, _>::new(provider)
        .run("x")
        .await
        .unwrap();
    assert_eq!(out.note, None);

    let format = seen.lock().unwrap()[0].1["response_format"].clone();
    assert_eq!(format["json_schema"]["strict"], false);
    assert_eq!(
        format["json_schema"]["schema"]["required"],
        json!(["number", "total_cents"])
    );
}
