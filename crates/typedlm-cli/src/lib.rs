//! Command line for TypedLM evaluations.
//!
//! Two entry points share this crate:
//!
//! - [`Evals`], a harness for a small binary in your project. It knows your signatures, so
//!   it can run them — on several models at once — and keeps every run as a snapshot.
//! - The `typedlm` binary (`cargo install typedlm-cli`), which works on stored reports
//!   only: show, compare, history. No project build needed, e.g. in CI.
//!
//! ```ignore
//! // src/bin/evals.rs
//! use typedlm::eval::ExactMatch;
//! use typedlm::http::OpenAiCompatible;
//!
//! fn main() -> std::process::ExitCode {
//!     typedlm_cli::Evals::new(OpenAiCompatible::ollama)
//!         .model("qwen3:32b")
//!         .program::<ClassifyTicket, _>("classify", "datasets/tickets.jsonl", ExactMatch)
//!         .run()
//! }
//! ```
//!
//! ```sh
//! cargo run --bin evals -- run classify --model qwen3:32b --model llama3.3
//! cargo run --bin evals -- compare classify
//! cargo run --bin evals -- history classify
//! ```

pub mod render;
pub mod store;

use std::ffi::OsString;
use std::future::Future;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::ExitCode;
use std::sync::Arc;

use lexopt::ValueExt;
use runemark::{ColorMode, Console};
use typedlm::eval::{Dataset, EvalOptions, Metric, Report, Verdict, compare, evaluate_with};
use typedlm::{DynProvider, Program, Provider, Signature};

pub use store::{DEFAULT_ROOT, Store};

/// Exit code for a significant regression.
pub const EXIT_REGRESSION: i32 = 1;
/// Exit code for wrong usage or unreadable files.
pub const EXIT_ERROR: i32 = 2;

type BoxProvider = Box<dyn DynProvider>;
type RunFuture = Pin<Box<dyn Future<Output = Result<Report, String>>>>;
type RunFn = Box<dyn Fn(BoxProvider, EvalOptions) -> RunFuture>;

struct Registered {
    name: String,
    run: RunFn,
}

/// The project-side harness: register signatures with their datasets, then call
/// [`run`](Evals::run) from `main`.
pub struct Evals {
    provider: Box<dyn Fn(&str) -> BoxProvider>,
    models: Vec<String>,
    programs: Vec<Registered>,
    store: Store,
}

impl Evals {
    /// `provider` builds a provider for a model name, e.g. `OpenAiCompatible::ollama` or
    /// `|model| OpenAiCompatible::new(url, model).api_key(key)`.
    pub fn new<P, F>(provider: F) -> Self
    where
        P: Provider + 'static,
        F: Fn(&str) -> P + 'static,
    {
        Self {
            provider: Box::new(move |model| Box::new(provider(model)) as BoxProvider),
            models: Vec::new(),
            programs: Vec::new(),
            store: Store::new(DEFAULT_ROOT),
        }
    }

    /// A model used when `run` gets no `--model`; may be called several times.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.models.push(model.into());
        self
    }

    /// Where snapshots go; default `.typedlm/evaluations`.
    pub fn store(mut self, root: impl Into<PathBuf>) -> Self {
        self.store = Store::new(root);
        self
    }

    /// Registers a signature with its dataset (JSON Lines) and metric.
    pub fn program<S, M>(self, name: &str, dataset: impl Into<PathBuf>, metric: M) -> Self
    where
        S: Signature + serde::de::DeserializeOwned + Clone + 'static,
        S::Output: serde::Serialize,
        M: Metric<S> + 'static,
    {
        self.program_with::<S, M>(name, dataset, metric, |program| program)
    }

    /// Like [`program`](Evals::program), with a function that configures the program,
    /// e.g. `|p| p.temperature(0.0).max_repairs(1)`.
    pub fn program_with<S, M>(
        mut self,
        name: &str,
        dataset: impl Into<PathBuf>,
        metric: M,
        configure: impl Fn(Program<S, BoxProvider>) -> Program<S, BoxProvider> + 'static,
    ) -> Self
    where
        S: Signature + serde::de::DeserializeOwned + Clone + 'static,
        S::Output: serde::Serialize,
        M: Metric<S> + 'static,
    {
        let dataset = dataset.into();
        let metric = Arc::new(metric);
        let configure = Arc::new(configure);
        let run: RunFn = Box::new(move |provider, options| {
            let dataset = dataset.clone();
            let metric = Arc::clone(&metric);
            let configure = Arc::clone(&configure);
            Box::pin(async move {
                let data = Dataset::<S>::from_jsonl(&dataset).map_err(|e| e.to_string())?;
                let program = configure(Program::new(provider));
                Ok(evaluate_with(&program, &data, metric.as_ref(), options).await)
            })
        });
        self.programs.push(Registered {
            name: name.into(),
            run,
        });
        self
    }

    /// Parses the process arguments, runs the command and returns its exit code.
    pub fn run(self) -> ExitCode {
        let console = Console::stdout(ColorMode::Auto);
        let code = self.run_with(std::env::args_os().skip(1), &mut std::io::stdout(), console);
        ExitCode::from(code as u8)
    }

    /// Like [`run`](Evals::run) with explicit arguments and output, for tests.
    pub fn run_with<I>(&self, args: I, out: &mut dyn Write, console: Console) -> i32
    where
        I: IntoIterator<Item = OsString>,
    {
        match self.dispatch(args.into_iter().collect(), out, console) {
            Ok(code) => code,
            Err(message) => {
                let _ = writeln!(out, "error: {message}");
                EXIT_ERROR
            }
        }
    }

    fn dispatch(
        &self,
        args: Vec<OsString>,
        out: &mut dyn Write,
        console: Console,
    ) -> Result<i32, String> {
        let mut parser = lexopt::Parser::from_args(args);
        let command = match parser.next().map_err(|e| e.to_string())? {
            Some(lexopt::Arg::Value(v)) => v.string().map_err(|e| e.to_string())?,
            Some(lexopt::Arg::Long("help") | lexopt::Arg::Short('h')) | None => {
                write!(out, "{HARNESS_USAGE}").map_err(io)?;
                return Ok(0);
            }
            Some(other) => return Err(other.unexpected().to_string()),
        };
        match command.as_str() {
            "list" => {
                for program in &self.programs {
                    writeln!(out, "{}", program.name).map_err(io)?;
                }
                Ok(0)
            }
            "run" => self.cmd_run(&mut parser, out),
            "compare" => {
                let args = CompareArgs::parse(&mut parser, true)?;
                let program = args.program.clone().ok_or("compare needs a program name")?;
                let history: Vec<(PathBuf, Report)> = self
                    .store
                    .history(&program)?
                    .into_iter()
                    .filter(|(_, r)| {
                        args.model.is_none() || store::run_name(r) == args.model.as_deref()
                    })
                    .collect();
                let Some((_, current)) = history.last() else {
                    return Err(format!(
                        "no snapshots for {program} in {}",
                        self.store.root().display()
                    ));
                };
                let (baseline_path, baseline) = match &args.baseline {
                    Some(path) => (path.clone(), store::read_report(path)?),
                    None => history
                        .len()
                        .checked_sub(2)
                        .map(|i| history[i].clone())
                        .ok_or_else(|| {
                            format!("{program} has only one snapshot; run it again first")
                        })?,
                };
                compare_reports(&baseline_path, &baseline, current, &args, console, out)
            }
            "history" => {
                let program = next_value(&mut parser)?.ok_or("history needs a program name")?;
                let entries = self.store.history(&program)?;
                if entries.is_empty() {
                    writeln!(out, "no snapshots for {program}").map_err(io)?;
                } else {
                    write!(out, "{}", render::history(&entries)).map_err(io)?;
                }
                Ok(0)
            }
            "help" => {
                write!(out, "{HARNESS_USAGE}").map_err(io)?;
                Ok(0)
            }
            other => Err(format!("unknown command {other:?}; see --help")),
        }
    }

    fn cmd_run(&self, parser: &mut lexopt::Parser, out: &mut dyn Write) -> Result<i32, String> {
        let mut names = Vec::new();
        let mut models = Vec::new();
        let mut options = EvalOptions::default();
        let mut save = true;
        while let Some(arg) = parser.next().map_err(|e| e.to_string())? {
            match arg {
                lexopt::Arg::Value(v) => names.push(v.string().map_err(|e| e.to_string())?),
                lexopt::Arg::Long("model") | lexopt::Arg::Short('m') => {
                    models.push(string(parser)?)
                }
                lexopt::Arg::Long("epochs") => options.epochs = number(parser)?,
                lexopt::Arg::Long("concurrency") => options.concurrency = number(parser)?,
                lexopt::Arg::Long("no-save") => save = false,
                other => return Err(other.unexpected().to_string()),
            }
        }
        if models.is_empty() {
            models = self.models.clone();
        }
        if models.is_empty() {
            return Err("no model: pass --model or register one with Evals::model".into());
        }
        let selected: Vec<&Registered> = if names.is_empty() {
            self.programs.iter().collect()
        } else {
            names
                .iter()
                .map(|n| {
                    self.programs
                        .iter()
                        .find(|p| &p.name == n)
                        .ok_or_else(|| format!("unknown program {n:?}; see `list`"))
                })
                .collect::<Result<_, _>>()?
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        for program in selected {
            let mut reports = Vec::new();
            for model in &models {
                writeln!(out, "{} on {model} …", program.name).map_err(io)?;
                let mut report =
                    runtime.block_on((program.run)((self.provider)(model), options))?;
                report.label = Some(model.clone());
                writeln!(out, "{report}").map_err(io)?;
                if let Some(line) = self.against_previous(&program.name, &report)? {
                    writeln!(out, "{line}").map_err(io)?;
                }
                if save {
                    let path = self.store.save(&program.name, &report)?;
                    writeln!(out, "saved {}", path.display()).map_err(io)?;
                }
                writeln!(out).map_err(io)?;
                reports.push(report);
            }
            if reports.len() > 1 {
                write!(out, "{}", render::matrix(&reports)).map_err(io)?;
            }
        }
        Ok(0)
    }

    /// One line comparing with the latest snapshot of the same program and model.
    fn against_previous(&self, program: &str, report: &Report) -> Result<Option<String>, String> {
        let previous = self
            .store
            .history(program)?
            .into_iter()
            .rev()
            .find(|(_, r)| store::run_name(r) == store::run_name(report));
        Ok(previous.and_then(|(path, before)| {
            let comparison = compare(&before, report, 0.0).ok()?;
            let file = path.file_name()?.to_string_lossy().into_owned();
            Some(format!("vs {file}: {comparison}"))
        }))
    }
}

const HARNESS_USAGE: &str = "\
Run and compare TypedLM evaluations.

Commands:
  list                              Registered programs
  run [PROGRAM...] [options]        Evaluate (all programs by default)
      -m, --model MODEL             Model to use; repeat for a comparison matrix
      --epochs N                    Runs per example (default 1)
      --concurrency N               Calls in flight (default 4)
      --no-save                     Do not store a snapshot
  compare PROGRAM [options]         Latest snapshot against the one before
      -m, --model MODEL             Only snapshots of this model
      --baseline FILE               Compare against this report instead
      --tolerance POINTS            Drop tolerated before a regression, e.g. 0.02
      --details                     List every changed example
  history PROGRAM                   All snapshots of a program

Exit codes: 0 ok, 1 significant regression, 2 error.
";

pub(crate) const REPORT_USAGE: &str = "\
Show and compare stored TypedLM reports.

Commands:
  show REPORT                       Print a report
  compare BASELINE CURRENT          Paired comparison of two reports
      --tolerance POINTS            Drop tolerated before a regression, e.g. 0.02
      --details                     List every changed example
  history DIR | PROGRAM             Snapshots in DIR, or of PROGRAM under --root
      --root DIR                    Snapshot root (default .typedlm/evaluations)

Exit codes: 0 ok, 1 significant regression, 2 error.
";

/// The `typedlm` binary: commands on stored reports.
pub fn report_tool<I>(args: I, out: &mut dyn Write, console: Console) -> i32
where
    I: IntoIterator<Item = OsString>,
{
    let result = (|| -> Result<i32, String> {
        let mut parser = lexopt::Parser::from_args(args);
        let command = match parser.next().map_err(|e| e.to_string())? {
            Some(lexopt::Arg::Value(v)) => v.string().map_err(|e| e.to_string())?,
            Some(lexopt::Arg::Long("help") | lexopt::Arg::Short('h')) | None => {
                write!(out, "{REPORT_USAGE}").map_err(io)?;
                return Ok(0);
            }
            Some(other) => return Err(other.unexpected().to_string()),
        };
        match command.as_str() {
            "show" => {
                let path = next_value(&mut parser)?.ok_or("show needs a report file")?;
                let report = store::read_report(Path::new(&path))?;
                writeln!(out, "{report}").map_err(io)?;
                Ok(0)
            }
            "compare" => {
                let args = CompareArgs::parse(&mut parser, false)?;
                let [baseline, current] = &args.files[..] else {
                    return Err("compare needs two report files: BASELINE CURRENT".into());
                };
                let (baseline_path, current_path) =
                    (PathBuf::from(baseline), PathBuf::from(current));
                let baseline = store::read_report(&baseline_path)?;
                let current = store::read_report(&current_path)?;
                compare_reports(&baseline_path, &baseline, &current, &args, console, out)
            }
            "history" => {
                let mut target = None;
                let mut root = PathBuf::from(DEFAULT_ROOT);
                while let Some(arg) = parser.next().map_err(|e| e.to_string())? {
                    match arg {
                        lexopt::Arg::Value(v) => {
                            target = Some(v.string().map_err(|e| e.to_string())?)
                        }
                        lexopt::Arg::Long("root") => root = PathBuf::from(string(&mut parser)?),
                        other => return Err(other.unexpected().to_string()),
                    }
                }
                let target = target.ok_or("history needs a directory or program name")?;
                let entries = if Path::new(&target).is_dir() {
                    store::read_dir(Path::new(&target))?
                } else {
                    Store::new(root).history(&target)?
                };
                if entries.is_empty() {
                    writeln!(out, "no snapshots for {target}").map_err(io)?;
                } else {
                    write!(out, "{}", render::history(&entries)).map_err(io)?;
                }
                Ok(0)
            }
            "help" => {
                write!(out, "{REPORT_USAGE}").map_err(io)?;
                Ok(0)
            }
            other => Err(format!("unknown command {other:?}; see --help")),
        }
    })();
    match result {
        Ok(code) => code,
        Err(message) => {
            let _ = writeln!(out, "error: {message}");
            EXIT_ERROR
        }
    }
}

struct CompareArgs {
    program: Option<String>,
    files: Vec<String>,
    model: Option<String>,
    baseline: Option<PathBuf>,
    tolerance: f64,
    details: bool,
}

impl CompareArgs {
    /// `harness`: first value is a program name and `--model`/`--baseline` apply;
    /// otherwise values are report files.
    fn parse(parser: &mut lexopt::Parser, harness: bool) -> Result<Self, String> {
        let mut args = Self {
            program: None,
            files: Vec::new(),
            model: None,
            baseline: None,
            tolerance: 0.0,
            details: false,
        };
        while let Some(arg) = parser.next().map_err(|e| e.to_string())? {
            match arg {
                lexopt::Arg::Value(v) => {
                    let v = v.string().map_err(|e| e.to_string())?;
                    if harness && args.program.is_none() {
                        args.program = Some(v);
                    } else if harness {
                        return Err(format!("unexpected argument {v:?}"));
                    } else {
                        args.files.push(v);
                    }
                }
                lexopt::Arg::Long("tolerance") => args.tolerance = number(parser)?,
                lexopt::Arg::Long("details") => args.details = true,
                lexopt::Arg::Long("model") | lexopt::Arg::Short('m') if harness => {
                    args.model = Some(string(parser)?)
                }
                lexopt::Arg::Long("baseline") if harness => {
                    args.baseline = Some(PathBuf::from(string(parser)?))
                }
                other => return Err(other.unexpected().to_string()),
            }
        }
        Ok(args)
    }
}

fn compare_reports(
    baseline_path: &Path,
    baseline: &Report,
    current: &Report,
    args: &CompareArgs,
    console: Console,
    out: &mut dyn Write,
) -> Result<i32, String> {
    let comparison = compare(baseline, current, args.tolerance).map_err(|e| e.to_string())?;
    let program = args.program.as_deref().unwrap_or(&current.program);
    let text = render::comparison(console, program, baseline_path, &comparison, args.details);
    write!(out, "{text}").map_err(io)?;
    Ok(match comparison.verdict {
        Verdict::Regression => EXIT_REGRESSION,
        _ => 0,
    })
}

fn next_value(parser: &mut lexopt::Parser) -> Result<Option<String>, String> {
    match parser.next().map_err(|e| e.to_string())? {
        Some(lexopt::Arg::Value(v)) => v.string().map(Some).map_err(|e| e.to_string()),
        Some(other) => Err(other.unexpected().to_string()),
        None => Ok(None),
    }
}

fn string(parser: &mut lexopt::Parser) -> Result<String, String> {
    parser
        .value()
        .map_err(|e| e.to_string())?
        .string()
        .map_err(|e| e.to_string())
}

fn number<T: std::str::FromStr>(parser: &mut lexopt::Parser) -> Result<T, String> {
    let text = string(parser)?;
    text.parse().map_err(|_| format!("not a number: {text:?}"))
}

fn io(e: std::io::Error) -> String {
    e.to_string()
}
