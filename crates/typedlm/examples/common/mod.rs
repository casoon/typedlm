//! Provider selection shared by the examples.
//!
//! - `TYPEDLM_MODEL` (required): model name, e.g. `qwen3:32b` or `gpt-5-mini`.
//! - `TYPEDLM_API_KEY` set: OpenAI-compatible API at `TYPEDLM_BASE_URL`
//!   (default `https://api.openai.com/v1`).
//! - otherwise: local Ollama at `TYPEDLM_BASE_URL` (default `http://localhost:11434/v1`).

use typedlm::http::OpenAiCompatible;

pub fn provider() -> OpenAiCompatible {
    let env = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    let Some(model) = env("TYPEDLM_MODEL") else {
        eprintln!("set TYPEDLM_MODEL, e.g. TYPEDLM_MODEL=qwen3:32b for a local Ollama");
        std::process::exit(2);
    };
    match env("TYPEDLM_API_KEY") {
        Some(key) => {
            let base_url = env("TYPEDLM_BASE_URL").unwrap_or("https://api.openai.com/v1".into());
            OpenAiCompatible::new(base_url, model).api_key(key)
        }
        None => match env("TYPEDLM_BASE_URL") {
            Some(base_url) => {
                // Same settings as `ollama()`, at another address.
                let capabilities = typedlm::Provider::capabilities(&OpenAiCompatible::ollama(""));
                OpenAiCompatible::new(base_url, model).capabilities(capabilities)
            }
            None => OpenAiCompatible::ollama(model),
        },
    }
}
