---
title: Tracing
description: What each call records through the tracing crate, and how to include inputs and answers.
order: 4
---

Every `Program::run` and `Program::execute` runs inside a `typedlm.execute` span at level INFO.
Field names follow the OpenTelemetry GenAI conventions where they exist.

| Field | Value |
| --- | --- |
| `typedlm.program` | name of the signature |
| `gen_ai.operation.name` | `chat` |
| `typedlm.strategy` | `native_schema`, `tool_call`, `json_mode` or `prompt_only` |
| `gen_ai.response.model` | model that answered, on success |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | sum over all attempts |
| `typedlm.attempts` | number of model calls |
| `typedlm.outcome` | `valid`, `repaired` or `failed` |
| `error.type` | `Error::kind()`, on failure |

Each attempt adds a DEBUG event `typedlm attempt` with `typedlm.attempt`, `typedlm.latency_ms`,
token usage and `typedlm.result` (`ok` or the error kind).

## Inputs and answers

They are not recorded by default, because they often contain personal data. Enable them per
program where your traces may hold them:

```rust
let classify = Program::<ClassifyTicket, _>::new(provider).record_content(true);
```

The input and each answer then appear as DEBUG events with the fields `typedlm.input` and
`typedlm.response`.

Any `tracing` subscriber works, for example `tracing-subscriber` for local output or an
OpenTelemetry exporter.
