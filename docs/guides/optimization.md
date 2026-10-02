---
title: Optimization
description: Let the optimizer choose worked examples and improve instructions against a validation set, and get the result as a compiled program.
order: 8
---

Which worked examples help a program, and which instructions, is hard to guess. The optimizer
(feature `optimize`) tries alternatives against a validation dataset and keeps what scores best:

- **worked examples** — `optimize_few_shot` chooses demonstrations from a pool,
- **instructions** — `optimize_instructions` has a teacher model rewrite them from the program's
  mistakes.

Both keep everything else about the program and return a compiled program.

## Candidates

The optimizer chooses from a pool of candidate demonstrations:

```rust
use typedlm::optimize::{bootstrap_demonstrations, demonstrations_from_labels};

// Labelled examples with a complete expected output:
let pool = demonstrations_from_labels(&train);

// Or the program's own correct answers — also with partial labels, and with a
// stronger model as teacher for a smaller one:
let teacher = Program::<ClassifyTicket, _>::new(OpenAiCompatible::new(url, "gpt-5").api_key(key));
let pool = bootstrap_demonstrations(&teacher, &train, &ExactMatch).await;
```

`bootstrap_demonstrations` runs the teacher on every training example and keeps the answers the
metric scores 1.

## Search

```rust
use typedlm::optimize::{FewShot, optimize_few_shot};

let options = FewShot { k: 4, max_calls: 400, seed: 0, concurrency: 4 };
let result = optimize_few_shot(&program, &pool, &validation, &ExactMatch, options).await?;

println!("{}", result.comparison);
result.compiled.save("classify.typedlm.json")?;
```

- The search starts from a seeded choice of `k` demonstrations and swaps one at a time, keeping a
  swap when the validation score rises. It stops when no swap helps (`local_optimum`) or the budget
  is spent.
- `max_calls` is the budget in model calls. Each evaluation costs one call per validation example
  (plus repairs), so 400 calls on a 20-example validation set allow 20 evaluations.
- No set is evaluated twice. The same seed gives the same run.
- The program's instructions, strategy and generation settings are kept; only its demonstrations
  change.

The search is pathwise's budgeted local search, built for evaluations that are expensive and noisy.

## Result

`Optimized` contains:

- `compiled` — the program with the best demonstrations, with that run as provenance; ready to
  [save and load](../compiled-programs/),
- `baseline` and `best` — reports of the program as given and of the winner on the validation set,
- `comparison` — paired comparison of the two; `Verdict::NoSignificantChange` means the data does
  not show that the demonstrations help,
- `trials` — every evaluated set with its score, in order.

## Instructions

```rust
use typedlm::optimize::{Reflective, optimize_instructions};

let teacher = OpenAiCompatible::new(url, "gpt-5").api_key(key);
let options = Reflective { iterations: 6, max_calls: 400, mistakes_shown: 5, concurrency: 4 };
let result = optimize_instructions(&program, &teacher, &validation, &ExactMatch, options).await?;
```

Each round:

1. The program's mistakes on the validation set — input, expected output, actual output or error
   — go to the teacher, up to `mistakes_shown`, rotating so each round sees others.
2. The teacher proposes complete new instructions: general rules, not quotes of the examples.
   The teacher is itself a TypedLM program (`ProposeInstructions` → `ProposedInstructions`), so its
   answer is validated like any other.
3. The proposal is scored on the validation set and kept if it scores higher.

The run stops after `iterations` proposals, when the budget is spent, or when nothing is wrong any
more. `trials` lists every proposal with its score; a failed proposal keeps its error. If no
proposal beats the original, the compiled program keeps the original instructions.

The teacher sees validation inputs and the program's answers. Choose a teacher those may be sent
to.

To optimize both, run `optimize_instructions`, load its compiled program, then run
`optimize_few_shot` on it — or the other way round.

## Validation is not a test

The winner is chosen because it scored best on the validation set, so its score there is
optimistic. Keep three sets apart:

| Set | Used for |
| --- | --- |
| train | candidate demonstrations |
| validation | choosing among them |
| test | the number you report, with `evaluate` or a [baseline](../evaluation/#regression-tests) |

## Requirements

`optimize` builds on [pathwise](https://github.com/casoon/pathwise), which needs Rust 1.88; the
rest of TypedLM supports 1.85.
