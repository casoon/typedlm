---
title: Evaluation
description: Datasets with partial labels, metrics, and a report you can trust only as far as its interval.
order: 3
---

## Datasets

One JSON object per line, with the input and the expected output:

```jsonl
{"input": {"text": "Our production database is down."}, "expected": {"urgency": "High", "category": "Technical"}}
{"input": {"text": "I was charged twice this month."}, "expected": {"category": "Billing"}}
```

`expected` may label only some top-level fields; unlabelled fields are not scored. Each line is
checked against the output type when loading, so wrong values and misspelt field names fail with
the line number:

```text
line 2: expected.urgncy: field is not allowed
```

```rust
use typedlm::eval::{Dataset, Example};

let dataset = Dataset::<ClassifyTicket>::from_jsonl("tickets.jsonl")?;

// Or in code, with a complete expected output:
let example = Example::new(ClassifyTicket { text: "…".into() }, &expected);
```

The input type needs `Deserialize` and `Clone` for evaluation.

## Metrics

| Metric | Score per example |
| --- | --- |
| `ExactMatch` | 1 if every labelled field is right, else 0 |
| `FieldAccuracy` | share of labelled fields that are right |

Numbers compare by value (`2` equals `2.0`). Your own metric implements `Metric<S>` and returns a
score between 0 and 1.

## Running an evaluation

```rust
use typedlm::eval::{ExactMatch, evaluate};

let report = evaluate(&classify, &dataset, &ExactMatch, 4).await; // at most 4 calls at once
println!("{report}");
```

Failed runs score 0. The report contains:

- the mean score with a 95 % confidence interval (Wilson interval for 0/1 scores, normal
  approximation otherwise),
- accuracy per labelled field,
- how many answers were valid at once, needed repair, or failed — with the error per failure,
- latency P50 and P95, token usage including failed runs, and the model that answered most often.

It holds no model output — only scores per example and a fingerprint of the dataset —
and serialises with serde in both directions.

## Several runs per example

Models answer differently from run to run, even at temperature 0. Run each example several times
to see how stable a program is:

```rust
use typedlm::eval::{EvalOptions, evaluate_with};

let options = EvalOptions { concurrency: 4, epochs: 3 };
let report = evaluate_with(&classify, &dataset, &ExactMatch, options).await;
```

The score per example becomes the mean over its runs, and the report counts the examples whose
runs all scored the same (`consistent`). Valid, repaired and failed are counted per run; tokens
and latency cover every run.

## Reading the interval

Eight examples at 87.5 % give an interval from 53 % to 98 %. A difference between two runs means
something only when the intervals barely overlap. Grow the dataset before drawing conclusions.

## Regression tests

A stored report is a baseline. Later runs are compared with it example by example, which
detects a real change with far fewer examples than comparing two averages.

```rust
use typedlm::eval::{Baseline, ExactMatch, evaluate};

#[tokio::test]
#[ignore = "calls a model"]
async fn classifier_has_not_regressed() {
    let report = evaluate(&classify, &dataset, &ExactMatch, 4).await;
    report.require_score(0.9).unwrap();
    Baseline::at("tests/baselines/classify.json").check(&report).unwrap();
}
```

- The first run writes the baseline; commit it.
- Later runs fail only on a **significant** drop: the whole 95 % interval of the
  per-example difference lies below zero (or below `-tolerance`, set with
  `Baseline::tolerance(0.02)` for two points).
- The error lists which examples got worse:

```text
exact match 100.0 % → 60.0 % (-40.0 points, 95 % CI -63.5 – -16.5) on 20 examples, 8 worse, 0 better: regression; worse examples: 0, 1, 5, 6, 10, 11, 15, 16
```

- After a deliberate change — another model, new instructions — accept the new state with
  `TYPEDLM_UPDATE_BASELINES=1 cargo test …`.
- A baseline only compares with a run on the same examples: reports carry a SHA-256 of
  the dataset, and a changed dataset is rejected instead of compared.
- `require_score(minimum)` fails only when even the upper end of the interval is below the
  minimum, so a small dataset does not fail by chance.

`compare(&baseline, &current, tolerance)` returns the same `Comparison` without files, for
your own tooling.

## Keeping evaluations out of the default test run

Evaluations call a real model: they cost money, need the network and vary between runs.
Keep them out of the default `cargo test`, for example behind `#[ignore]` or an
environment variable, and run them deliberately before changing model or instructions.
