---
title: Overview
description: What TypedLM does, where it stops, and how this documentation is organised.
order: 0
---

TypedLM treats a language model call as a function with a typed contract. The input is a Rust
struct, the output is an ordinary Rust type, and every answer is validated against it before your
code sees it. The same contract describes a dataset, so the program can be measured like any
other component.

## What it covers

- **Signatures**: `#[derive(TypedLm)]` on the input struct, any `Deserialize + JsonSchema` type
  as output, doc comments as instructions.
- **Programs**: binding a signature to a provider, choosing how structured output is obtained,
  repairing invalid answers.
- **Validation**: tolerant JSON extraction, schema validation with every violation and its path,
  your own domain rules.
- **Evaluation**: datasets with partial labels, metrics, a report with confidence interval.
- **Tracing**: one span per call with OpenTelemetry GenAI field names.

## Where it stops

TypedLM is not an agent framework. There is no orchestration of multiple agents, no workflow
engine, no vector store, no RAG pipeline and no memory. It answers one question: how to define,
test and improve a single LLM-based building block. Systems that need more can use TypedLM for
their individual calls.

## Status

Early development, not published on crates.io. The API will change before 0.1. Tested live against
a local Ollama with `qwen3:32b`; OpenAI has not been tested live yet.

## This documentation

- **Getting started**: installation and a first program.
- **Guides**: providers, validation, evaluation, tracing.
- **Concepts**: how the pieces fit together and why they are built that way.
- **Project**: how to work on TypedLM itself.
