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

It holds no model output and serialises with serde, e.g. to store it next to the dataset.

## Reading the interval

Eight examples at 87.5 % give an interval from 53 % to 98 %. A difference between two runs means
something only when the intervals barely overlap. Grow the dataset before drawing conclusions.

## In tests

Evaluations call a real model: they cost money, need the network and vary between runs. Keep them
out of the default `cargo test`, for example behind `#[ignore]` or an environment variable, and
compare against a threshold with room for the interval.
