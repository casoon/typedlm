---
title: Installation
description: Add TypedLM from GitHub, pick the features you need, and check the requirements.
order: 1
---

```sh
cargo add typedlm
cargo add serde --features derive
cargo add schemars
cargo add tokio --features macros,rt-multi-thread
```

Your own types derive `Serialize`, `Deserialize` and `JsonSchema`, so `serde` and `schemars` are
direct dependencies of your crate. Any async runtime works; the examples use tokio.

## Features

| Feature | Default | Adds |
| --- | --- | --- |
| `derive` | yes | `#[derive(TypedLm)]` |
| `http` | yes | `OpenAiCompatible`, using reqwest with rustls and the tokio timer |
| `eval` | yes | `Dataset`, metrics and `evaluate` |
| `optimize` | no | optimizing worked examples and instructions |

Without default features the crate depends on `serde`, `serde_json`, `schemars` and `tracing` only,
and on no async runtime. Bring your own `Provider` implementation in that case.

## Requirements

| | Minimum |
| --- | --- |
| Rust | 1.85 (edition 2024); 1.88 with the `optimize` feature |
| Endpoint | any OpenAI-compatible chat completions API, or your own `Provider` |
