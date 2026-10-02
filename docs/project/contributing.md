---
title: Contributing
description: Conventions and checks for working on TypedLM itself.
order: 1
---

## Checks

`scripts/verify.sh` must pass before every push. It runs:

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-features`
- `cargo check -p typedlm --no-default-features` — the core builds without HTTP, derive and eval
- `cargo doc` with warnings as errors
- `cargo +1.85 check` — the minimum supported Rust version

CI runs the same on pull requests and pushes to `main`.

## Live tests

Tests against real endpoints are `#[ignore]` and live in `crates/typedlm/tests/live.rs`:

```sh
TYPEDLM_LIVE_BASE_URL=http://localhost:11434/v1 TYPEDLM_LIVE_MODEL=qwen3:32b \
TYPEDLM_LIVE_DIALECT=ollama cargo test --test live -- --ignored --nocapture
```

`TYPEDLM_LIVE_API_KEY` is optional; `TYPEDLM_LIVE_DIALECT` is `strict` (default), `generic` or
`ollama`; `TYPEDLM_LIVE_STRATEGIES` limits the strategies, e.g. `tool_call,json_mode`.

## Conventions

- **Dependencies**: only what users barely pay for; write narrow pieces instead of adding heavy or
  rare crates. No `thiserror`, no `async-trait`. Keep `syn` on the major version serde's and
  schemars' derives use, so it is not compiled twice.
- **Errors**: own enums, `#[non_exhaustive]`, `Display` and `Error` by hand. `Display` never prints
  model output.
- **Public enums** that may grow are `#[non_exhaustive]`.
- **Async**: native `async fn` in traits; object safety through a separate `Dyn…` trait with a
  blanket impl.
- **Runtime**: the core does not depend on an async runtime; `tokio` only in the `http` feature and
  in tests.
- **Names**: acronyms as words (`TypedLm`); derive attributes under `#[lm(...)]`; descriptions from
  doc comments.
- **Tracing**: OpenTelemetry GenAI field names, otherwise `typedlm.*`; no inputs or answers without
  opt-in.
- **Tests**: offline and deterministic. Use the providers in `typedlm::testing`; HTTP tests run
  against a small server inside the test, not a mock crate.
