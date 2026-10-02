//! `typedlm`: show, compare and list stored evaluation reports.

use std::process::ExitCode;

use runemark::{ColorMode, Console};

fn main() -> ExitCode {
    let console = Console::stdout(ColorMode::Auto);
    let code =
        typedlm_cli::report_tool(std::env::args_os().skip(1), &mut std::io::stdout(), console);
    ExitCode::from(code as u8)
}
