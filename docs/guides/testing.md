---
title: Testing without a model
description: Fixed answers, closures, schema-valid answers, and recording real interactions to replay them offline.
order: 5
---

Code around a program — the `match` on the output, error handling, retries — should be tested in
the normal `cargo test`, without network or costs. `typedlm::testing` provides providers for that.
None of them adds a dependency.

## Fixed answers

```rust
use typedlm::testing::TestProvider;

let provider = TestProvider::answers([r#"{"urgency": "High"}"#, r#"{"urgency": "Low"}"#]);
let classify = Program::<ClassifyTicket, _>::new(&provider);

assert_eq!(classify.run("db down").await?.urgency, Urgency::High);
assert_eq!(provider.requests()[0].input["text"], "db down");
```

Answers are used in order; one request more than answers is an error. Pass the provider by
reference (`&provider`) to inspect `requests()` afterwards — every provider works behind `&` and
`Arc`.

Answers go through the same validation as real ones, so an invalid answer exercises the repair
path: `TestProvider::answers(["nope", r#"{"urgency": "Low"}"#])` makes `execute` report two
attempts.

## Any valid answer

```rust
let provider = TestProvider::valid();
```

Generates an answer from the output schema: required fields only, the first enum variant, the
smallest allowed number, empty lists. Useful when the content does not matter, only that the code
around the program runs.

## Answers from a closure

```rust
use typedlm::testing::FnProvider;

let provider = FnProvider::new(|request| {
    let urgent = request.input["text"].as_str().unwrap_or("").contains("down");
    Ok(format!(r#"{{"urgency": "{}"}}"#, if urgent { "High" } else { "Low" }))
});
```

The closure sees the whole `Request` and returns the answer text or a `ProviderError`, e.g. to
test how your code handles a provider failure.

## Recording and replaying real interactions

Record once against a real model, then replay offline in every test run:

```rust
use typedlm::testing::{Recorder, Replay};

// Once, with network and key:
let recorder = Recorder::new("tests/recordings/classify.json", OpenAiCompatible::ollama("qwen3:32b"))?;
Program::<ClassifyTicket, _>::new(&recorder).run("db down").await?;

// In every test:
let replay = Replay::open("tests/recordings/classify.json")?;
let ticket = Program::<ClassifyTicket, _>::new(replay).run("db down").await?;
```

- The file stores each request with its response, keyed by a SHA-256 of the request. Nothing else
  is stored: no API keys, no headers. It holds inputs and model answers, so treat it like test
  data.
- `Replay` reports the recorded provider's capabilities, so the program builds exactly the same
  requests.
- A request that was not recorded fails with `ProviderError::Recording`. Changing instructions,
  input, schema or generation options changes the request, so a stale recording shows up as a
  failing test instead of silently passing.
- Recording again keeps earlier interactions and overwrites the ones it repeats.

Combined with a [baseline](../evaluation/#regression-tests), a recorded evaluation runs offline:
record the dataset once, then compare replayed runs in CI.
