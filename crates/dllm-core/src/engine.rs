//! `Engine` trait + `MockEngine` for Phase 0.
//!
//! Real inference (llama-cpp-2) lands in Phase 1 behind a Cargo feature gate.

use tokio::sync::mpsc;

/// One streamed generation event.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TokenEvent {
    /// Zero-based position in this generation.
    pub pos: u32,
    /// Text for this step (word for `MockEngine`).
    pub text: String,
    /// True on the final event of the stream.
    pub done: bool,
}

/// Inference engine abstraction.
///
/// `generate_stream` is synchronous and returns a channel receiver; the
/// engine spawns the generation task internally. Callers just drain `rx`.
pub trait Engine: Send + Sync + 'static {
    /// Names of locally available models (e.g. `qwen3-0.6b-q4`).
    fn model_list(&self) -> Vec<String>;

    /// Start a generation; tokens arrive on the returned channel.
    fn generate_stream(&self, prompt: String) -> mpsc::Receiver<TokenEvent>;
}

/// Phase 0 mock: streams a canned sentence word-by-word with 30 ms delays.
#[derive(Debug, Clone, Default)]
pub struct MockEngine {
    models: Vec<String>,
}

impl MockEngine {
    pub fn new() -> Self {
        Self {
            models: vec!["qwen3-0.6b-q4".to_string()],
        }
    }

    pub fn with_models(models: Vec<String>) -> Self {
        Self { models }
    }
}

impl Engine for MockEngine {
    fn model_list(&self) -> Vec<String> {
        self.models.clone()
    }

    fn generate_stream(&self, _prompt: String) -> mpsc::Receiver<TokenEvent> {
        let (tx, rx) = mpsc::channel(32);
        tokio::spawn(async move {
            let canned = "Hello from the distributed LAN mesh streaming one word at a time.";
            let words: Vec<&str> = canned.split_whitespace().collect();
            let n = words.len() as u32;
            for (i, word) in words.into_iter().enumerate() {
                tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                let done = (i as u32) + 1 == n;
                let ev = TokenEvent {
                    pos: i as u32,
                    text: word.to_string(),
                    done,
                };
                if tx.send(ev).await.is_err() {
                    break;
                }
                if done {
                    break;
                }
            }
        });
        rx
    }
}
