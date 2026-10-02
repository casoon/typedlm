---
title: Design decisions
description: The decisions TypedLM is built on, why they hold, and what they mean for its code.
order: 2
---

## A building block, not an agent framework

TypedLM covers single LLM-based components. There is no multi-agent orchestration, workflow
engine, RAG pipeline, vector store or memory. Systems that need those can use TypedLM for their
individual calls. Features that require multi-step autonomous behaviour belong in another project.

## Ordinary Rust types

Inputs and outputs are plain types with `serde` and `schemars` derives; TypedLM adds no type
system of its own. The output type can be reused outside LLM code, datasets are typed against it,
and nothing is generated that you cannot see in your editor. The input struct carries the derive,
the output is a separate type — not one struct with input and output fields.

## The model proposes, Rust validates

No code path hands unvalidated model output to the caller as a typed value. Every answer passes
schema validation, deserialisation and the signature's own rules.

## Provider-independent core

The core knows the `Provider` trait only. Vendor specifics — request format, schema dialect,
retries — live in provider implementations. One OpenAI-compatible provider is built in; other
vendors are reached through OpenAI-compatible routers rather than a provider each.

## Small dependency surface

Dependencies that nearly every Rust project with LLM code already has (serde, schemars, syn,
tracing, reqwest) are used. Narrow pieces are written instead of pulling in heavy or rare crates:
the schema validator, tolerant JSON extraction, the confidence interval. The core has no async
runtime dependency.

## No model output in logs by default

Error messages and traces carry metadata only. Raw answers stay in error fields, and inputs and
answers enter traces only on request.

## Optimisation separate from runtime

Planned: optimising instructions and examples produces an artefact that the production runtime
loads. The optimiser will never be a dependency of the runtime.
