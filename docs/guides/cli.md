---
title: Command line
description: Run programs across models, keep every run as a snapshot, and compare runs — from your project or on stored reports.
order: 6
---

`typedlm-cli` has two entry points:

- **A harness in your project.** It knows your signatures, so it can run them — on several models
  at once — and keeps every run as a snapshot.
- **The `typedlm` binary** (`cargo install typedlm-cli`). It works on stored reports only: show, compare, history. No project
  build needed, e.g. in CI.

## The harness

Add a small binary to your project:

```rust
// src/bin/evals.rs
use typedlm::eval::ExactMatch;
use typedlm::http::OpenAiCompatible;

fn main() -> std::process::ExitCode {
    typedlm_cli::Evals::new(OpenAiCompatible::ollama)
        .model("qwen3:32b")
        .program::<ClassifyTicket, _>("classify", "datasets/tickets.jsonl", ExactMatch)
        .run()
}
```

`Evals::new` takes a function from a model name to a provider, e.g.
`|model| OpenAiCompatible::new(url, model).api_key(key)`. Use `program_with` to configure the
program: `.program_with::<ClassifyTicket, _>("classify", path, ExactMatch, |p| p.temperature(0.0))`.

```sh
cargo run --bin evals -- list
cargo run --bin evals -- run                       # every program, default model
cargo run --bin evals -- run classify --model qwen3:32b --model llama3.3 --epochs 3
cargo run --bin evals -- compare classify          # latest snapshot against the one before
cargo run --bin evals -- compare classify --model qwen3:32b --tolerance 0.02
cargo run --bin evals -- history classify
```

`run` prints the report for each model, a line comparing it with that model's previous snapshot,
and — with several models — a matrix:

```text
model      exact match          95 % CI    valid  repaired  failed        p95  tokens
─────────────────────────────────────────────────────────────────────────────────────
qwen3:32b       87.5 %  52.9 % – 97.8 %  100.0 %     0.0 %   0.0 %  121426 ms    3024
```

Every run is stored as `.typedlm/evaluations/<program>/<UTC time>-<model>.json` (`--no-save` to
skip, `Evals::store` for another directory). The file name sorts in run order. Commit the
directory to keep the history with the code, or ignore it.

Runs are labelled with the requested model name, so `--model` filters work even when the provider
reports a dated model snapshot.

## The `typedlm` binary

```sh
cargo install typedlm-cli

typedlm show .typedlm/evaluations/classify/2026-10-02T10-03-47.555Z-qwen3-32b.json
typedlm compare baseline.json current.json --tolerance 0.02
typedlm history classify                     # under .typedlm/evaluations, or --root DIR
typedlm history path/to/snapshots/
```

## Comparisons

`compare` pairs the two runs example by example (see
[regression tests](../evaluation/#regression-tests)) and prints a verdict, the scores with the
interval of their difference, and the examples that got worse or better (the first few;
`--details` lists all). Runs on different datasets are refused.

Exit codes: `0` ok, `1` significant regression, `2` wrong usage or unreadable files — so
`typedlm compare` can gate a CI step.
