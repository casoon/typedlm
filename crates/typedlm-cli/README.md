# typedlm-cli

Command line for [TypedLM](https://crates.io/crates/typedlm) evaluations.

- **Harness for your project** (`typedlm_cli::Evals`): a small binary in your project that
  knows your signatures — run programs across several models as a matrix, keep every run
  as a snapshot, compare runs, show history.
- **`typedlm` binary** for stored reports: `show`, `compare` (exit code 1 on a significant
  regression), `history`.

```sh
cargo install typedlm-cli
typedlm compare baseline.json current.json
```

Guide: <https://casoon.github.io/typedlm/docs/guides/cli/>
