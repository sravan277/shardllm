//! Pipeline planning: layer ranges, KV budgets, calibration.
//!
//! Adopted splits (research 00-index):
//! - Balanced default: 9/10/9  => `0-8 / 9-18 / 19-27`.
//! - Heterogeneity fit only: 16/8/4 => `0-15 / 16-23 / 24-27`.

/// Inclusive layer range `[start, end]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StageRange {
    pub start: u32,
    pub end: u32,
}

impl StageRange {
    pub fn new(start: u32, end: u32) -> Self {
        assert!(start <= end, "StageRange start must be <= end");
        Self { start, end }
    }

    /// Number of layers in this range.
    pub fn len(&self) -> u32 {
        self.end - self.start + 1
    }

    pub fn is_empty(&self) -> bool {
        false // inclusive range always holds >= 1 layer
    }

    /// KV budget in bytes for `ctx_tokens` of context.
    ///
    /// Assumes [`KV_BYTES_PER_TOKEN_PER_LAYER`] (4 KiB/layer/token,
    /// i.e. ~112 KiB/token over 28 layers).
    pub fn kv_budget_bytes(&self, ctx_tokens: u32) -> u64 {
        self.len() as u64 * KV_BYTES_PER_TOKEN_PER_LAYER as u64 * ctx_tokens as u64
    }
}

/// Total decoder layers for Qwen3-0.6B.
pub const TOTAL_LAYERS: u32 = 28;
/// Default 4K context window.
pub const CTX_TOKENS_4K: u32 = 4096;
/// 4 KiB per layer per token (FP16 first; 112 KiB/token over 28 layers).
pub const KV_BYTES_PER_TOKEN_PER_LAYER: usize = 4 * 1024;
/// Total KV per token across all 28 layers (~112 KiB).
pub const KV_BYTES_PER_TOKEN_TOTAL: usize = 112 * 1024;
/// Total KV for 4K context (~448 MiB).
pub const KV_BUDGET_4K_TOTAL_BYTES: u64 =
    KV_BYTES_PER_TOKEN_TOTAL as u64 * CTX_TOKENS_4K as u64;

/// Per-stage KV budgets @4K for the balanced 9/10/9 split.
pub const KV_BUDGET_STAGE_A_9_LAYERS_4K: u64 =
    9 * CTX_TOKENS_4K as u64 * KV_BYTES_PER_TOKEN_PER_LAYER as u64;
pub const KV_BUDGET_STAGE_B_10_LAYERS_4K: u64 =
    10 * CTX_TOKENS_4K as u64 * KV_BYTES_PER_TOKEN_PER_LAYER as u64;
pub const KV_BUDGET_STAGE_C_9_LAYERS_4K: u64 =
    9 * CTX_TOKENS_4K as u64 * KV_BYTES_PER_TOKEN_PER_LAYER as u64;

/// Per-stage KV budgets @4K for the hetero 16/8/4 split.
pub const KV_BUDGET_STAGE_A_16_LAYERS_4K: u64 =
    16 * CTX_TOKENS_4K as u64 * KV_BYTES_PER_TOKEN_PER_LAYER as u64;
pub const KV_BUDGET_STAGE_B_8_LAYERS_4K: u64 =
    8 * CTX_TOKENS_4K as u64 * KV_BYTES_PER_TOKEN_PER_LAYER as u64;
pub const KV_BUDGET_STAGE_C_4_LAYERS_4K: u64 =
    4 * CTX_TOKENS_4K as u64 * KV_BYTES_PER_TOKEN_PER_LAYER as u64;

/// Immutable pipeline assignment: ordered stage ranges + id.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PipelinePlan {
    pub plan_id: u64,
    pub stages: Vec<StageRange>,
}

impl PipelinePlan {
    /// Balanced 9/10/9 (default; research 03 recommendation).
    pub fn balanced_9_10_9() -> Self {
        Self {
            plan_id: 1,
            stages: vec![
                StageRange::new(0, 8),
                StageRange::new(9, 18),
                StageRange::new(19, 27),
            ],
        }
    }

    /// Heterogeneity fit 16/8/4 (documented fallback only).
    pub fn hetero_16_8_4() -> Self {
        Self {
            plan_id: 2,
            stages: vec![
                StageRange::new(0, 15),
                StageRange::new(16, 23),
                StageRange::new(24, 27),
            ],
        }
    }

    pub fn num_layers(&self) -> u32 {
        self.stages.iter().map(|s| s.len()).sum()
    }

    /// KV budget per stage @4K, in bytes.
    pub fn kv_budgets_4k(&self) -> Vec<u64> {
        self.stages
            .iter()
            .map(|s| s.kv_budget_bytes(CTX_TOKENS_4K))
            .collect()
    }
}

/// Measured per-device cost model from the pairing calibration bench.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Calibration {
    /// Milliseconds per layer for batch-1 decode.
    pub ms_per_layer_decode: f64,
    /// Milliseconds per layer for a prefill chunk.
    pub ms_per_layer_prefill_chunk: f64,
    /// Network hop cost (ms) to the next stage.
    pub hop_ms: f64,
}

impl Calibration {
    pub fn stage_time_decode(&self, num_layers: u32) -> f64 {
        self.ms_per_layer_decode * num_layers as f64 + self.hop_ms
    }

    pub fn stage_time_prefill(&self, num_layers: u32) -> f64 {
        self.ms_per_layer_prefill_chunk * num_layers as f64 + self.hop_ms
    }
}

/// Greedy layer partition across stages.
///
/// Allocates layers proportional to `1 / ms_per_layer_decode` (faster
/// devices get more layers), ensuring contiguous coverage, every layer
/// assigned exactly once, and >= 1 layer per stage.
pub fn greedy_partition(total_layers: u32, stages: &[Calibration]) -> Vec<StageRange> {
    assert!(!stages.is_empty(), "need at least one stage");
    assert!(
        total_layers as usize >= stages.len(),
        "more stages than layers"
    );

    let weights: Vec<f64> = stages
        .iter()
        .map(|c| {
            if c.ms_per_layer_decode <= 0.0 || !c.ms_per_layer_decode.is_finite() {
                1.0
            } else {
                1.0 / c.ms_per_layer_decode
            }
        })
        .collect();
    let wsum: f64 = weights.iter().sum();

    let mut counts: Vec<u32> = weights
        .iter()
        .map(|w| ((w / wsum) * total_layers as f64).floor() as u32)
        .collect();
    for c in counts.iter_mut() {
        if *c == 0 {
            *c = 1;
        }
    }

    // Fix rounding so counts sum to total_layers.
    let mut sum: u32 = counts.iter().sum();
    let mut i = 0usize;
    let n = counts.len();
    while sum < total_layers {
        counts[i % n] += 1;
        sum += 1;
        i += 1;
    }
    let mut guard = 0usize;
    while sum > total_layers && guard < 100_000 {
        if let Some((idx, _)) = counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 1)
            .max_by_key(|(_, &c)| c)
        {
            counts[idx] -= 1;
            sum -= 1;
        } else {
            break;
        }
        guard += 1;
    }

    let mut out = Vec::with_capacity(counts.len());
    let mut start = 0u32;
    for c in counts {
        let end = start + c - 1;
        out.push(StageRange::new(start, end));
        start = end + 1;
    }
    out
}
