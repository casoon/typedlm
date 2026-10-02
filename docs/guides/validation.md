---
title: Validation and repair
description: What happens between the model's answer and your typed output, and how to add your own rules.
order: 2
---

## The pipeline

Every answer goes through the same steps; each one has its own error.

1. **Finish reason**: a refusal becomes `Error::Refusal`, a cut-off answer `Error::Truncated`.
   Neither is repaired.
2. **JSON extraction**: code fences, text around the JSON and trailing commas are tolerated. An
   unclosed value is reported, never completed. Failure: `Error::InvalidJson`.
3. **Schema validation**: against the schema generated from the output type. Every violation is
   collected with its path, e.g. `$.items[2].quantity: expected a value <= 255, got 300`.
   Failure: `Error::SchemaViolation`.
4. **Deserialization** into the output type. Failure: `Error::Deserialize`.
5. **Your rules** from `#[lm(validate = …)]`. Failure: `Error::Validation`.

## Repair

When a step from 2 to 5 fails, the program sends the answer back with a description of what is
wrong — all violations at once — and asks for a corrected answer. `Program::max_repairs` sets how
often (default 2). With 0, the first error is returned as is. When all attempts fail, the result is
`Error::RetryLimitExceeded { attempts }` with every attempt's error, the last one last.

## Your own rules

Rules the schema cannot express go into a function. Its messages are sent to the model, so phrase
them as instructions.

```rust
/// Extract the invoice data from the text. Amounts are in cents.
#[derive(TypedLm, Serialize)]
#[lm(output = Invoice, validate = items_add_up)]
struct ExtractInvoice {
    text: String,
}

fn items_add_up(invoice: &Invoice) -> Result<(), Vec<String>> {
    let items: u64 = invoice.items.iter().map(|i| u64::from(i.quantity) * i.unit_price_cents).sum();
    if items != invoice.net_cents {
        return Err(vec![format!(
            "the line items add up to {items} cents but net_cents is {}; check quantities and unit prices",
            invoice.net_cents
        )]);
    }
    Ok(())
}
```

Numeric ranges and string lengths are schema keywords; express them with `schemars` attributes
(`#[schemars(range(min = 0, max = 100))]`) and the validator checks them.

## Errors and personal data

Variants such as `InvalidJson` and `SchemaViolation` keep the model's raw answer in a field.
`Display` never prints model output — not even values quoted in a violation — so logging an error
does not leak it. `Error::kind()` returns a stable name for metrics, e.g. `schema_violation`.

## Without a program

`parse_output::<S>(&response, &output_schema::<S>())` runs the pipeline on a single `Response`,
for example in tests or in your own provider loop.
