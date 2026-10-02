---
title: Compiled programs
description: Store a program's instructions, worked examples and settings as a reviewable JSON artefact, and load it in production with a check against the signature.
order: 7
---

A program's behaviour depends on more than its types: instructions, worked examples, the
strategy and generation settings. Once a configuration is good — measured, not guessed — store it
as a **compiled program**: one JSON file next to the code, reviewed in pull requests like any other
change. Production loads it; whatever produced it does not need to ship.

## Worked examples

A few demonstrations often help more than longer instructions, especially with small models:

```rust
let classify = Program::<ClassifyTicket, _>::new(provider)
    .instructions("Decide how urgent the ticket is. Outages are High.")
    .demonstration("Checkout is down", &TicketClassification { urgency: Urgency::High, category: Category::Technical })
    .demonstration("How do I change my avatar?", &TicketClassification { urgency: Urgency::Low, category: Category::Other })
    .temperature(0.0);
```

Each demonstration is sent before the input as a user turn with its input and an assistant turn
with its output.

## Compile

```rust
let report = evaluate(&classify, &dataset, &ExactMatch, 4).await;
classify.compile().with_provenance(&report).save("classify.typedlm.json")?;
```

```json
{
  "format": 1,
  "program": "ClassifyTicket",
  "signature": "…64 hex digits…",
  "instructions": "Decide how urgent the ticket is. Outages are High.",
  "demonstrations": [
    { "input": { "text": "Checkout is down" }, "output": { "urgency": "High", "category": "Technical" } }
  ],
  "strategy": null,
  "generation": { "temperature": 0.0, "max_tokens": null },
  "max_repairs": 2,
  "provenance": {
    "model": "qwen3:32b",
    "metric": "exact match",
    "score": 0.875,
    "interval": [0.529, 0.978],
    "examples": 8,
    "dataset": "…"
  }
}
```

- `signature` is a SHA-256 over the signature's name and output schema.
- `provenance` records the evaluation the configuration was accepted on: model, metric, score with
  interval, number of examples and the dataset's fingerprint. It is optional.
- The provider is not part of the artefact: the same file runs on any model you point it at.

## Load

```rust
use typedlm::CompiledProgram;

let classify = CompiledProgram::load("classify.typedlm.json")?
    .program::<ClassifyTicket, _>(OpenAiCompatible::ollama("qwen3:32b"))?;
```

Loading checks the artefact against the signature:

- A different signature or a changed output type is refused with `CompileError::SignatureMismatch`
  — compile the program again.
- Every demonstration must fit the input type and the output schema; a hand-edited mistake fails
  with `CompileError::InvalidDemonstration` and its index.
- A file from a newer format version is refused.

## Embed it

Compile the artefact into the binary and check it in a test, so a mismatch fails the build
pipeline, not production:

```rust
const CLASSIFY: &str = include_str!("../classify.typedlm.json");

pub fn classifier<P: Provider>(provider: P) -> Result<Program<ClassifyTicket, P>, CompileError> {
    CompiledProgram::from_json(CLASSIFY)?.program(provider)
}

#[test]
fn compiled_classifier_fits_the_signature() {
    classifier(typedlm::testing::TestProvider::valid()).unwrap();
}
```

## What comes next

Today a compiled program records a configuration you tuned by hand. An optimiser that searches
instructions and worked examples against a dataset will produce the same artefact.
