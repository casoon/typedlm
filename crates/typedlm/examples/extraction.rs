//! Extract a nested invoice structure and check it against domain rules.
//!
//! ```sh
//! TYPEDLM_MODEL=qwen3:32b cargo run --example extraction
//! ```

mod common;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::prelude::*;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct LineItem {
    description: String,
    quantity: u32,
    /// Net unit price in cents.
    unit_price_cents: u64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Invoice {
    number: String,
    /// Issue date as YYYY-MM-DD.
    date: String,
    items: Vec<LineItem>,
    /// Net total in cents, before tax.
    net_cents: u64,
    /// Grand total in cents, including tax.
    total_cents: u64,
    /// Anything unusual about the invoice, if mentioned.
    note: Option<String>,
}

/// Extract the invoice data from the text. Amounts are in cents.
#[derive(TypedLm, Serialize)]
#[lm(output = Invoice, validate = items_add_up)]
struct ExtractInvoice {
    text: String,
}

/// Rules the schema cannot express. Failing ones are sent back to the model,
/// so each message says what to fix.
fn items_add_up(invoice: &Invoice) -> Result<(), Vec<String>> {
    let items: u64 = invoice
        .items
        .iter()
        .map(|item| u64::from(item.quantity) * item.unit_price_cents)
        .sum();
    let mut errors = Vec::new();
    if items != invoice.net_cents {
        errors.push(format!(
            "the line items add up to {items} cents but net_cents is {}; check quantities and unit prices",
            invoice.net_cents
        ));
    }
    if invoice.total_cents < invoice.net_cents {
        errors.push("total_cents must not be smaller than net_cents".into());
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

const TEXT: &str = "Invoice no. RE-2026-0042, issued 14 September 2026.
2 x Server maintenance at 150.00 EUR each
1 x Emergency call-out at 89.50 EUR
Subtotal 389.50 EUR, VAT 19 % 74.01 EUR, total 463.51 EUR.
Payable within 14 days.";

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let extract = Program::<ExtractInvoice, _>::new(common::provider()).temperature(0.0);
    let invoice = extract.run(TEXT).await?;

    println!("{} from {}", invoice.number, invoice.date);
    for item in &invoice.items {
        println!(
            "  {} x {} à {:.2}",
            item.quantity,
            item.description,
            item.unit_price_cents as f64 / 100.0
        );
    }
    println!(
        "net {:.2}, total {:.2}",
        invoice.net_cents as f64 / 100.0,
        invoice.total_cents as f64 / 100.0
    );
    if let Some(note) = invoice.note {
        println!("note: {note}");
    }
    Ok(())
}
