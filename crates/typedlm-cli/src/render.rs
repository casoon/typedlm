//! Terminal output for reports, model matrices, comparisons and history.

use std::path::Path;

use runemark::{
    Console, DetailLevel, Finding, FindingGroup, Metric, ScopeNote, Tone, Trend, Verdict,
};
use typedlm::eval::{Comparison, Report, Verdict as Change};

fn pct(x: f64) -> String {
    format!("{:.1} %", x * 100.0)
}

fn share(n: usize, report: &Report) -> String {
    let runs = report.examples * report.epochs.max(1);
    pct(if runs == 0 {
        0.0
    } else {
        n as f64 / runs as f64
    })
}

/// Fixed-width table with left-aligned first column and right-aligned others.
fn table(header: &[&str], rows: &[Vec<String>]) -> String {
    let widths: Vec<usize> = (0..header.len())
        .map(|c| {
            rows.iter()
                .map(|r| r[c].chars().count())
                .chain([header[c].chars().count()])
                .max()
                .unwrap_or(0)
        })
        .collect();
    let line = |cells: Vec<&str>| {
        let parts: Vec<String> = cells
            .iter()
            .enumerate()
            .map(|(c, cell)| {
                if c == 0 {
                    format!("{cell:<w$}", w = widths[c])
                } else {
                    format!("{cell:>w$}", w = widths[c])
                }
            })
            .collect();
        parts.join("  ").trim_end().to_string()
    };
    let mut out = line(header.to_vec());
    out.push('\n');
    out.push_str(&"─".repeat(widths.iter().sum::<usize>() + 2 * (widths.len() - 1)));
    for row in rows {
        out.push('\n');
        out.push_str(&line(row.iter().map(String::as_str).collect()));
    }
    out.push('\n');
    out
}

/// One row per model, for runs of the same program on the same dataset.
pub fn matrix(reports: &[Report]) -> String {
    let rows: Vec<Vec<String>> = reports
        .iter()
        .map(|r| {
            vec![
                crate::store::run_name(r).unwrap_or("?").to_string(),
                pct(r.score),
                format!("{} – {}", pct(r.interval.0), pct(r.interval.1)),
                share(r.valid, r),
                share(r.repaired, r),
                share(r.failed, r),
                format!("{} ms", r.latency_p95_ms),
                format!("{}", r.usage.input_tokens + r.usage.output_tokens),
            ]
        })
        .collect();
    let metric = reports.first().map_or("score", |r| r.metric.as_str());
    table(
        &[
            "model", metric, "95 % CI", "valid", "repaired", "failed", "p95", "tokens",
        ],
        &rows,
    )
}

/// `2026-10-02T09-02-03Z-model` → `2026-10-02 09:02`; other names unchanged.
fn snapshot_time(stem: &str) -> String {
    match (
        stem.get(..10),
        stem.get(10..11),
        stem.get(11..13),
        stem.get(14..16),
    ) {
        (Some(date), Some("T"), Some(hours), Some(minutes)) => {
            format!("{date} {hours}:{minutes}")
        }
        _ => stem.to_string(),
    }
}

/// Snapshots oldest first; the date comes from the file name.
pub fn history(entries: &[(std::path::PathBuf, Report)]) -> String {
    let rows: Vec<Vec<String>> = entries
        .iter()
        .map(|(path, r)| {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            vec![
                snapshot_time(stem),
                crate::store::run_name(r).unwrap_or("?").to_string(),
                pct(r.score),
                format!("{} – {}", pct(r.interval.0), pct(r.interval.1)),
                format!("{}", r.examples),
                format!("{}", r.epochs.max(1)),
                share(r.failed, r),
            ]
        })
        .collect();
    let metric = entries.first().map_or("score", |(_, r)| r.metric.as_str());
    table(
        &[
            "date (UTC)",
            "model",
            metric,
            "95 % CI",
            "examples",
            "epochs",
            "failed",
        ],
        &rows,
    )
}

/// A comparison as a decision-oriented report: verdict, scores, worse examples.
pub fn comparison(
    console: Console,
    program: &str,
    baseline: &Path,
    comparison: &Comparison,
    details: bool,
) -> String {
    let (verdict, summary) = match comparison.verdict {
        Change::Regression => (Verdict::Failed, "significant regression"),
        Change::Improvement => (Verdict::Passed, "significant improvement"),
        Change::NoSignificantChange => (Verdict::Passed, "no significant change"),
    };
    let points = |x: f64| format!("{:+.1}", x * 100.0);
    let trend = match comparison.verdict {
        Change::Regression => Trend::Negative,
        Change::Improvement => Trend::Positive,
        Change::NoSignificantChange => Trend::Neutral,
    };
    let mut report = runemark::Report::new(format!("{program}: {summary}"), verdict)
        .add_metric(Metric::new("baseline", pct(comparison.baseline_score)))
        .add_metric(
            Metric::new("current", pct(comparison.current_score))
                .with_trend(trend, format!("{} points", points(comparison.difference))),
        )
        .add_metric(Metric::new(
            "95 % CI",
            format!(
                "{} – {} points",
                points(comparison.interval.0),
                points(comparison.interval.1)
            ),
        ))
        .add_metric(Metric::new("examples", comparison.examples.to_string()));

    for (title, indices, tone) in [
        ("Worse than baseline", &comparison.worse, Tone::Error),
        ("Better than baseline", &comparison.better, Tone::Success),
    ] {
        if indices.is_empty() {
            continue;
        }
        let mut group = FindingGroup::new(title);
        for index in indices {
            group = group.add_finding(Finding::new(tone, format!("example {index}")));
        }
        report = report.add_group(group);
    }
    if details {
        report = report.with_detail_level(DetailLevel::Detailed);
    }
    report
        .add_scope_note(ScopeNote::new(
            "Baseline",
            vec![baseline.display().to_string()],
        ))
        .render(console)
}

#[cfg(test)]
mod tests {
    use super::*;
    use typedlm::Usage;

    /// The recorded evaluation from examples/recorded/evaluation.txt.
    fn recorded() -> Report {
        Report {
            program: "ClassifyTicket".into(),
            dataset: String::new(),
            model: Some("qwen3:32b".into()),
            label: Some("qwen3:32b".into()),
            metric: "exact match".into(),
            examples: 8,
            epochs: 1,
            consistent: 8,
            score: 0.875,
            interval: (0.529_105_194_230_138_6, 0.977_583_091_136_703_8),
            fields: Vec::new(),
            valid: 8,
            repaired: 0,
            failed: 0,
            failures: Vec::new(),
            scores: vec![1.0; 8],
            latency_p50_ms: 96_248,
            latency_p95_ms: 121_426,
            usage: Usage {
                input_tokens: 1099,
                output_tokens: 1925,
            },
        }
    }

    #[test]
    fn matrix_layout() {
        // Same text as in docs/guides/cli.md.
        let expected = concat!(
            "model      exact match          95 % CI    valid  repaired  failed        p95  tokens\n",
            "─────────────────────────────────────────────────────────────────────────────────────\n",
            "qwen3:32b       87.5 %  52.9 % – 97.8 %  100.0 %     0.0 %   0.0 %  121426 ms    3024\n",
        );
        assert_eq!(matrix(&[recorded()]), expected);
    }
}
