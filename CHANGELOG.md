# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
