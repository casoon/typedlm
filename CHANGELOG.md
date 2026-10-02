# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-10-02

### Added

- `Signature` trait and `#[derive(TypedLm)]` with `#[lm(output, name, description, validate)]`.
- `Program` with strategy selection, repair of invalid answers, `run` and `execute`.
- Validation pipeline: tolerant JSON extraction, schema validation with paths, custom rules.
- `OpenAiCompatible` provider with four strategies, schema dialects and transport retries.
- Evaluation: datasets with partial labels, `ExactMatch`, `FieldAccuracy`, reports with
  confidence intervals.
- Tracing spans with OpenTelemetry GenAI field names.
- Regression tests: `Baseline` stores a report and compares later runs example by example,
  `compare` for paired comparisons, `Report::require_score` against the interval. Reports
  carry per-example scores and a dataset fingerprint and can be read back.
- `evaluate_with` and `EvalOptions`: several runs per example, with a count of examples whose
  runs agree.
- `typedlm::testing`: `TestProvider` (fixed or schema-valid answers, recorded requests),
  `FnProvider`, `Recorder` and `Replay` for offline tests.
- `Provider` for `&P` and `Arc<P>`; `Request`, `Response` and `Capabilities` serialise.
- `typedlm-cli`: the `Evals` harness (`run` across models with a matrix, `compare`, `history`,
  snapshots under `.typedlm/evaluations`) and the `typedlm` binary for stored reports.
- `Report::label` for the run's configuration, e.g. the requested model.
- Worked examples: `Program::demonstration`, sent before the input as user/assistant turns.
- Compiled programs: `Program::compile`, `CompiledProgram` (JSON artefact with signature hash and
  optional provenance), loading with checks of signature and demonstrations.
- `Program` implements `Debug` without showing the provider.

- Optimizer (feature `optimize`): `optimize_few_shot` chooses demonstrations by budgeted local
  search on a validation set; candidates from `demonstrations_from_labels` or
  `bootstrap_demonstrations`; result as `CompiledProgram` with a paired comparison.

- `optimize_instructions`: a teacher model proposes instructions from the program's mistakes
  on a validation set; better proposals are kept. The teacher is a TypedLM program itself
  (`ProposeInstructions`).

- Typed tools: action enums as output, `tool_specs` (one tool per variant, native tool calls with
  `tool_choice: required`), `Action`, `Policy` (writes need confirmation by default),
  `ToolProgram` and a `typedlm.tool` tracing span.

### Fixed

- Reports, baselines and artefacts read back numbers exactly (`serde_json` with
  `float_roundtrip`).
