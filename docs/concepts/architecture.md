---
title: Architecture
description: The parts of TypedLM, how a call flows through them, and where each one lives in the code.
order: 1
---

## Crates

| Crate | Content |
| --- | --- |
| `typedlm` | the library: signatures, programs, providers, validation, evaluation |
| `typedlm-macros` | `#[derive(TypedLm)]`, used through `typedlm` |
| `typedlm-cli` | `Evals` harness, snapshot store, the `typedlm` binary for stored reports |

Modules in `crates/typedlm/src/`:

| Module | Responsibility |
| --- | --- |
| `signature.rs` | the `Signature` trait |
| `program.rs` | `Program`: strategy choice, instructions, repair loop, tracing |
| `provider.rs` | `Provider`, `DynProvider`, `Request`, `Response`, `Capabilities` |
| `dialect.rs` | `schema_for_dialect`: schema rewriting per provider |
| `output.rs` | `parse_output`: the validation pipeline |
| `json.rs` | tolerant JSON extraction |
| `schema.rs` | schema validator, `Violation` |
| `error.rs` | `Error`, repair feedback |
| `http.rs` | `OpenAiCompatible` (feature `http`) |
| `eval.rs` | datasets, metrics, `evaluate`, `Report` (feature `eval`) |
| `eval/regression.rs` | `Baseline`, `compare`, `Report::require_score` |
| `sha256.rs` | dataset and request fingerprints |
| `testing.rs` | `TestProvider`, `FnProvider`, `Recorder`, `Replay` |

## Signature

`Signature` is implemented on the input type. It names the output type, which must implement
`DeserializeOwned + JsonSchema`, and carries a name and a description as constants. The derive
fills them from `#[lm(output = …)]` and the doc comment; implementing the trait by hand is
supported. An optional `validate` function holds domain rules.

## A call, step by step

1. `Program::execute` serialises the input to JSON.
2. It asks the provider for its `Capabilities` and chooses the strategy: the one set on the
   program, otherwise the strongest the provider supports.
3. It rewrites the output schema into the provider's dialect and builds the instructions: the
   description, plus the schema when the strategy or the provider requires it.
4. The provider sends the request and returns the raw answer, the model, the finish reason and
   token usage.
5. `parse_output` turns the answer into the output type or an error (see
   [Validation and repair](../../guides/validation/)).
6. A repairable error adds the answer and its feedback to the next request, until the answer is
   valid or the attempts are used up.
7. The result is an `Execution` (output, attempts, usage, model, strategy) or a `Failed` with the
   error, attempts and usage.

All of it runs inside one tracing span.

## Providers

`Provider` uses native `async fn` in traits, so a program over a concrete provider has no boxing.
`DynProvider` is the object-safe counterpart, implemented for every provider; `Box<dyn
DynProvider>` is a provider again. Transport retries live in the provider, repairs in the program.

## Evaluation

`evaluate` runs every example through `Program::execute` with a concurrency limit
(`futures-util`, no runtime binding) and builds a `Report` from the outcomes. It reuses the
schema validator to check dataset labels when loading. The report keeps each example's
score and a SHA-256 of the dataset, so `eval/regression.rs` can compare two runs pairwise and
refuse runs on different examples.
