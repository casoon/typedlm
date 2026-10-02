//! Evaluation snapshots on disk: one JSON report per run under
//! `<root>/<program>/<UTC timestamp>-<model>.json`. Timestamps carry milliseconds and are
//! unique within a directory, so sorting file names gives the order of the runs.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use typedlm::eval::Report;

/// Default location, relative to the project root.
pub const DEFAULT_ROOT: &str = ".typedlm/evaluations";

#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Writes the report under `program` and returns its path. The file name ends with
    /// the report's label, or its model when it has none.
    pub fn save(&self, program: &str, report: &Report) -> Result<PathBuf, String> {
        let dir = self.root.join(slug(program));
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let model = run_name(report).map_or("unknown".into(), slug);
        let taken: Vec<String> = std::fs::read_dir(&dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .collect();
        // Two runs in the same millisecond: move the later one on, keeping names in order.
        let mut millis = now_millis();
        while taken
            .iter()
            .any(|name| name.starts_with(&utc_stamp(millis)))
        {
            millis += 1;
        }
        let path = dir.join(format!("{}-{model}.json", utc_stamp(millis)));
        write_report(&path, report)?;
        Ok(path)
    }

    /// Snapshots of `program`, oldest first.
    pub fn history(&self, program: &str) -> Result<Vec<(PathBuf, Report)>, String> {
        read_dir(&self.root.join(slug(program)))
    }
}

/// The label of a run, or the model that answered when there is none.
pub fn run_name(report: &Report) -> Option<&str> {
    report.label.as_deref().or(report.model.as_deref())
}

/// Reports in `dir`, oldest first (file names start with a UTC timestamp).
pub fn read_dir(dir: &Path) -> Result<Vec<(PathBuf, Report)>, String> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{}: {e}", dir.display())),
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|p| read_report(&p).map(|r| (p, r)))
        .collect()
}

pub fn read_report(path: &Path) -> Result<Report, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{} is not a report: {e}", path.display()))
}

pub fn write_report(path: &Path, report: &Report) -> Result<(), String> {
    let mut json = serde_json::to_string_pretty(report).expect("reports serialize");
    json.push('\n');
    std::fs::write(path, json).map_err(|e| format!("{}: {e}", path.display()))
}

/// File-name-safe form: letters, digits, `.`, `-` and `_`; everything else becomes `-`.
fn slug(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '-'
            }
        })
        .collect()
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// `YYYY-MM-DDTHH-MM-SS.mmmZ`: RFC 3339 in UTC with `-` instead of `:`, valid in file
/// names on every platform and sortable as text.
fn utc_stamp(millis: u64) -> String {
    let secs = millis / 1000;
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    let rem = secs % 86_400;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}-{:02}-{:02}.{:03}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60,
        millis % 1000
    )
}

/// Days since 1970-01-01 to (year, month, day), proleptic Gregorian
/// (H. Hinnant's `civil_from_days`).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamps() {
        assert_eq!(utc_stamp(0), "1970-01-01T00-00-00.000Z");
        assert_eq!(utc_stamp(951_782_400_000), "2000-02-29T00-00-00.000Z");
        assert_eq!(utc_stamp(1_790_931_723_042), "2026-10-02T09-02-03.042Z");
    }

    #[test]
    fn slugs() {
        assert_eq!(slug("qwen3:32b"), "qwen3-32b");
        assert_eq!(slug("openai/gpt-5-mini"), "openai-gpt-5-mini");
    }
}
