use std::ffi::OsString;
use std::path::{Path, PathBuf};

use runemark::{ColorMode, Console};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::eval::ExactMatch;
use typedlm::prelude::*;
use typedlm::testing::FnProvider;
use typedlm_cli::{EXIT_ERROR, EXIT_REGRESSION, Evals, report_tool};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
enum Urgency {
    Low,
    High,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Classification {
    urgency: Urgency,
}

/// Classify a support ticket.
#[derive(TypedLm, Serialize, Deserialize, Clone, Debug)]
#[lm(output = Classification)]
struct Classify {
    text: String,
}

/// Model "good" spots outages, model "lazy" always answers Low.
fn model(
    name: &str,
) -> FnProvider<impl Fn(&typedlm::Request) -> Result<String, typedlm::ProviderError> + use<>> {
    let good = name == "good";
    FnProvider::new(move |request| {
        let down = request.input["text"]
            .as_str()
            .unwrap_or("")
            .contains("down");
        let urgency = if good && down { "High" } else { "Low" };
        Ok(format!(r#"{{"urgency": "{urgency}"}}"#))
    })
}

fn workspace(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("typedlm-cli-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let lines: Vec<String> = (0..20)
        .map(|i| {
            let (text, urgency) = if i % 5 < 2 {
                ("service is down", "High")
            } else {
                ("question", "Low")
            };
            format!(
                r#"{{"input": {{"text": "{text} {i}"}}, "expected": {{"urgency": "{urgency}"}}}}"#
            )
        })
        .collect();
    std::fs::write(dir.join("tickets.jsonl"), lines.join("\n")).unwrap();
    dir
}

fn evals(dir: &Path) -> Evals {
    Evals::new(model)
        .model("good")
        .store(dir.join("snapshots"))
        .program_with::<Classify, _>("classify", dir.join("tickets.jsonl"), ExactMatch, |p| {
            p.temperature(0.0)
        })
}

fn run(evals: &Evals, args: &[&str]) -> (i32, String) {
    let mut out = Vec::new();
    let args: Vec<OsString> = args.iter().map(OsString::from).collect();
    let code = evals.run_with(args, &mut out, Console::new(ColorMode::Never, false));
    (code, String::from_utf8(out).unwrap())
}

fn tool(args: &[&str]) -> (i32, String) {
    let mut out = Vec::new();
    let args: Vec<OsString> = args.iter().map(OsString::from).collect();
    let code = report_tool(args, &mut out, Console::new(ColorMode::Never, false));
    (code, String::from_utf8(out).unwrap())
}

#[test]
fn run_saves_snapshots_and_prints_a_matrix_for_several_models() {
    let dir = workspace("matrix");
    let evals = evals(&dir);

    let (code, out) = run(&evals, &["run", "--model", "good", "--model", "lazy"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("classify on good …"), "{out}");
    assert!(out.contains("exact match     100.0 %"), "{out}");
    let matrix = out
        .lines()
        .skip_while(|l| !l.starts_with("model"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        matrix.contains("good") && matrix.contains("100.0 %"),
        "{matrix}"
    );
    assert!(
        matrix.contains("lazy") && matrix.contains("60.0 %"),
        "{matrix}"
    );

    let snapshots = std::fs::read_dir(dir.join("snapshots/Classify"))
        .unwrap()
        .count();
    assert_eq!(snapshots, 2);
}

#[test]
fn compare_finds_a_regression_between_runs() {
    let dir = workspace("compare");
    let evals = evals(&dir);
    assert_eq!(run(&evals, &["run"]).0, 0);

    let (code, out) = run(&evals, &["run", "--model", "good"]);
    assert_eq!(code, 0);
    assert!(out.contains(": exact match 100.0 % → 100.0 %"), "{out}");
    assert!(out.contains("no significant change"), "{out}");

    // Same program, worse model: the latest snapshot against the one before.
    assert_eq!(run(&evals, &["run", "--model", "lazy"]).0, 0);
    let (code, out) = run(&evals, &["compare", "classify"]);
    assert_eq!(code, EXIT_REGRESSION, "{out}");
    assert!(
        out.contains("[FAIL] classify: significant regression"),
        "{out}"
    );
    assert!(out.contains("example 0"), "{out}");

    // Restricted to one model there is no regression.
    let (code, out) = run(&evals, &["compare", "classify", "--model", "good"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("no significant change"), "{out}");

    let (code, out) = run(&evals, &["history", "classify"]);
    assert_eq!(code, 0);
    assert_eq!(
        out.lines()
            .filter(|l| l.contains("good") || l.contains("lazy"))
            .count(),
        3,
        "{out}"
    );
}

#[test]
fn report_tool_works_on_files() {
    let dir = workspace("tool");
    let evals = evals(&dir);
    run(&evals, &["run", "--model", "good"]);
    run(&evals, &["run", "--model", "lazy"]);
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir.join("snapshots/Classify"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    let (good, lazy) = (files[0].to_str().unwrap(), files[1].to_str().unwrap());

    let (code, out) = tool(&["show", good]);
    assert_eq!(code, 0);
    assert!(
        out.starts_with("Classify\nrun             good\nmodel           test"),
        "{out}"
    );

    assert_eq!(tool(&["compare", good, lazy]).0, EXIT_REGRESSION);
    assert_eq!(tool(&["compare", lazy, good]).0, 0);
    assert_eq!(tool(&["compare", good, lazy, "--tolerance", "0.7"]).0, 0);

    let (code, out) = tool(&["history", dir.join("snapshots/Classify").to_str().unwrap()]);
    assert_eq!(code, 0);
    assert!(out.contains("good") && out.contains("lazy"), "{out}");
}

#[test]
fn usage_errors_exit_with_two() {
    let dir = workspace("errors");
    let evals = evals(&dir);
    let (code, out) = run(&evals, &["run", "unknown"]);
    assert_eq!(code, EXIT_ERROR);
    assert!(out.contains("unknown program \"unknown\""), "{out}");
    assert_eq!(
        run(&evals, &["compare", "classify"]).0,
        EXIT_ERROR,
        "no snapshots yet"
    );
    assert_eq!(run(&evals, &["frobnicate"]).0, EXIT_ERROR);
    assert_eq!(tool(&["compare", "only-one.json"]).0, EXIT_ERROR);
    assert_eq!(run(&evals, &["--help"]).0, 0);
}
