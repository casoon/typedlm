//! Comparing an evaluation with a stored baseline, for regression tests.
//!
//! Both runs score the same examples, so the comparison is paired: per example, the
//! difference between the two scores. That detects a real change with far fewer examples
//! than comparing two means would.

use std::fmt;
use std::path::{Path, PathBuf};

use super::Report;

/// Environment variable that makes [`Baseline::check`] overwrite the stored baseline.
pub const UPDATE_BASELINES: &str = "TYPEDLM_UPDATE_BASELINES";

/// What a comparison concludes, at 95 % confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The current run is worse; the whole interval of the difference is below
    /// `-tolerance`.
    Regression,
    /// The current run is better; the whole interval is above zero.
    Improvement,
    /// The data cannot tell the runs apart.
    NoSignificantChange,
}

/// A paired comparison of two evaluations of the same dataset.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub metric: String,
    pub examples: usize,
    pub baseline_score: f64,
    pub current_score: f64,
    /// `current_score - baseline_score`.
    pub difference: f64,
    /// 95 % confidence interval of the difference.
    pub interval: (f64, f64),
    /// Indices of examples that scored lower than in the baseline.
    pub worse: Vec<usize>,
    /// Indices of examples that scored higher than in the baseline.
    pub better: Vec<usize>,
    pub verdict: Verdict,
}

/// Why a regression check failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum RegressionError {
    /// The current run is significantly worse than the baseline.
    Regression(Box<Comparison>),
    /// Even the upper end of the score's interval is below the required minimum.
    BelowMinimum {
        score: f64,
        interval: (f64, f64),
        minimum: f64,
    },
    /// The reports cover different examples, programs or metrics.
    NotComparable { reason: String },
    /// The baseline file could not be read or written.
    Io { path: PathBuf, message: String },
    /// The baseline file is not a valid report.
    InvalidBaseline { path: PathBuf, message: String },
}

/// Compares `current` with `baseline`. A drop counts as a regression only when it is
/// larger than `tolerance` (a score difference, e.g. `0.02` for two points) with 95 %
/// confidence.
pub fn compare(
    baseline: &Report,
    current: &Report,
    tolerance: f64,
) -> Result<Comparison, RegressionError> {
    let not_comparable = |reason: String| Err(RegressionError::NotComparable { reason });
    if baseline.program != current.program {
        return not_comparable(format!(
            "program {} differs from the baseline's {}",
            current.program, baseline.program
        ));
    }
    if baseline.metric != current.metric {
        return not_comparable(format!(
            "metric {} differs from the baseline's {}",
            current.metric, baseline.metric
        ));
    }
    if baseline.dataset != current.dataset || baseline.scores.len() != current.scores.len() {
        return not_comparable(
            "the dataset changed since the baseline was recorded; update the baseline".into(),
        );
    }

    let differences: Vec<f64> = current
        .scores
        .iter()
        .zip(&baseline.scores)
        .map(|(now, before)| now - before)
        .collect();
    let indices = |keep: fn(f64) -> bool| -> Vec<usize> {
        differences
            .iter()
            .enumerate()
            .filter(|(_, d)| keep(**d))
            .map(|(i, _)| i)
            .collect()
    };
    let difference = current.score - baseline.score;
    let interval = paired_interval(&differences);
    let verdict = if interval.1 < -tolerance {
        Verdict::Regression
    } else if interval.0 > 0.0 {
        Verdict::Improvement
    } else {
        Verdict::NoSignificantChange
    };

    Ok(Comparison {
        metric: current.metric.clone(),
        examples: differences.len(),
        baseline_score: baseline.score,
        current_score: current.score,
        difference,
        interval,
        worse: indices(|d| d < 0.0),
        better: indices(|d| d > 0.0),
        verdict,
    })
}

impl Report {
    /// Fails when even the upper end of the score's 95 % interval is below `minimum`,
    /// i.e. when the data shows the program misses the bar, not merely when this run's
    /// point estimate happens to.
    pub fn require_score(&self, minimum: f64) -> Result<(), RegressionError> {
        if self.interval.1 < minimum {
            return Err(RegressionError::BelowMinimum {
                score: self.score,
                interval: self.interval,
                minimum,
            });
        }
        Ok(())
    }
}

/// A report stored as JSON, for regression tests.
///
/// ```ignore
/// let report = evaluate(&program, &dataset, &ExactMatch, 4).await;
/// Baseline::at("tests/baselines/classify.json").check(&report)?;
/// ```
///
/// The first run writes the file. Later runs compare with it and fail on a significant
/// regression. Set `TYPEDLM_UPDATE_BASELINES=1` to accept the current run as the new
/// baseline, e.g. after a deliberate change of model or instructions.
#[derive(Debug, Clone)]
pub struct Baseline {
    path: PathBuf,
    tolerance: f64,
}

/// What [`Baseline::check`] did.
#[derive(Debug, Clone)]
pub enum BaselineOutcome {
    /// No baseline existed; the current report was stored.
    Created,
    /// `TYPEDLM_UPDATE_BASELINES` was set; the current report replaced the baseline.
    Updated,
    /// Compared with the baseline without a significant regression.
    Compared(Comparison),
}

impl Baseline {
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            tolerance: 0.0,
        }
    }

    /// Score drop tolerated before a significant change counts as regression
    /// (default 0).
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load(&self) -> Result<Report, RegressionError> {
        let text = std::fs::read_to_string(&self.path).map_err(|e| self.io(e))?;
        serde_json::from_str(&text).map_err(|e| RegressionError::InvalidBaseline {
            path: self.path.clone(),
            message: e.to_string(),
        })
    }

    pub fn store(&self, report: &Report) -> Result<(), RegressionError> {
        if let Some(parent) = self.path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| self.io(e))?;
        }
        let mut json = serde_json::to_string_pretty(report).expect("reports serialize to JSON");
        json.push('\n');
        std::fs::write(&self.path, json).map_err(|e| self.io(e))
    }

    pub fn check(&self, report: &Report) -> Result<BaselineOutcome, RegressionError> {
        let update = std::env::var(UPDATE_BASELINES).is_ok_and(|v| !v.is_empty() && v != "0");
        if update {
            self.store(report)?;
            return Ok(BaselineOutcome::Updated);
        }
        if !self.path.exists() {
            self.store(report)?;
            return Ok(BaselineOutcome::Created);
        }
        let comparison = compare(&self.load()?, report, self.tolerance)?;
        match comparison.verdict {
            Verdict::Regression => Err(RegressionError::Regression(Box::new(comparison))),
            _ => Ok(BaselineOutcome::Compared(comparison)),
        }
    }

    fn io(&self, error: std::io::Error) -> RegressionError {
        RegressionError::Io {
            path: self.path.clone(),
            message: error.to_string(),
        }
    }
}

/// 95 % interval of the mean of paired differences, with Student's t for small samples.
fn paired_interval(differences: &[f64]) -> (f64, f64) {
    let n = differences.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = differences.iter().sum::<f64>() / n as f64;
    if n == 1 {
        return (mean, mean);
    }
    let variance = differences.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let half = t_975(n - 1) * (variance / n as f64).sqrt();
    (mean - half, mean + half)
}

/// Two-sided 95 % quantile of Student's t distribution.
fn t_975(degrees_of_freedom: usize) -> f64 {
    const TABLE: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    match degrees_of_freedom {
        0 => f64::INFINITY,
        df @ 1..=30 => TABLE[df - 1],
        // Cornish-Fisher expansion; within 0.002 of the exact value above 30.
        df => {
            const Z: f64 = 1.959_964;
            Z + (Z.powi(3) + Z) / (4.0 * df as f64)
        }
    }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pct = |x: f64| format!("{:.1} %", x * 100.0);
        let points = |x: f64| format!("{:+.1}", x * 100.0);
        let verdict = match self.verdict {
            Verdict::Regression => "regression",
            Verdict::Improvement => "improvement",
            Verdict::NoSignificantChange => "no significant change",
        };
        write!(
            f,
            "{} {} → {} ({} points, 95 % CI {} – {}) on {} examples, {} worse, {} better: {verdict}",
            self.metric,
            pct(self.baseline_score),
            pct(self.current_score),
            points(self.difference),
            points(self.interval.0),
            points(self.interval.1),
            self.examples,
            self.worse.len(),
            self.better.len(),
        )
    }
}

impl fmt::Display for RegressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Regression(comparison) => {
                write!(f, "{comparison}")?;
                if !comparison.worse.is_empty() {
                    let list: Vec<String> = comparison.worse.iter().map(usize::to_string).collect();
                    write!(f, "; worse examples: {}", list.join(", "))?;
                }
                Ok(())
            }
            Self::BelowMinimum {
                score,
                interval,
                minimum,
            } => write!(
                f,
                "score {:.1} % (95 % CI {:.1} % – {:.1} %) is below the required {:.1} %",
                score * 100.0,
                interval.0 * 100.0,
                interval.1 * 100.0,
                minimum * 100.0
            ),
            Self::NotComparable { reason } => {
                write!(f, "cannot compare with the baseline: {reason}")
            }
            Self::Io { path, message } => write!(f, "{}: {message}", path.display()),
            Self::InvalidBaseline { path, message } => {
                write!(f, "{} is not a valid report: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for RegressionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_quantiles() {
        assert_eq!(t_975(7), 2.365);
        assert!((t_975(40) - 2.021).abs() < 0.002);
        assert!((t_975(1000) - 1.962).abs() < 0.002);
    }

    #[test]
    fn paired_interval_of_constant_differences_is_a_point() {
        assert_eq!(paired_interval(&[-1.0, -1.0, -1.0]), (-1.0, -1.0));
        assert_eq!(paired_interval(&[0.0; 5]), (0.0, 0.0));
    }

    #[test]
    fn paired_interval_matches_reference() {
        // mean -0.25, sd 0.4629, n 8, t(7) 2.365 → half width 0.3871
        let d = [-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0];
        let (low, high) = paired_interval(&d);
        assert!((low - (-0.6371)).abs() < 1e-3, "{low}");
        assert!((high - 0.1371).abs() < 1e-3, "{high}");
    }
}
