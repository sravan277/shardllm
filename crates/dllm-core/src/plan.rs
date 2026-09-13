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

// ---------------------------------------------------------------------------
// Calibration-driven planner (`DeviceSpec` + `plan_layers`).
// ---------------------------------------------------------------------------

/// Activation width on the wire (`contracts/catalog.json` `wire_dim`).
pub const WIRE_DIM: usize = 1024;
/// Bytes per wire element (fp16 activations).
pub const WIRE_BYTES_PER_ELEMENT: usize = 2;
/// Activation payload bytes per token per stage boundary (1024 fp16 = 2 KiB).
pub const ACTIVATION_BYTES_PER_TOKEN: usize = WIRE_DIM * WIRE_BYTES_PER_ELEMENT;

/// Per-device calibration input for [`plan_layers`].
///
/// `decode_tps` is the full-model batch-1 decode throughput measured at
/// pairing (reference: 16.6 tok/s in `docs/bench-baseline.json`);
/// `bandwidth_mbps` is the pairwise link bandwidth in megabits per second.
/// `kv_budget_mib` is informational (per-device KV budget, MiB); it is not
/// part of the partition objective yet.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeviceSpec {
    pub node_id: String,
    pub decode_tps: f64,
    pub bandwidth_mbps: f64,
    pub kv_budget_mib: u64,
}

impl DeviceSpec {
    pub fn new(
        node_id: impl Into<String>,
        decode_tps: f64,
        bandwidth_mbps: f64,
        kv_budget_mib: u64,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            decode_tps,
            bandwidth_mbps,
            kv_budget_mib,
        }
    }

    fn sane_tps(&self) -> f64 {
        if self.decode_tps.is_finite() && self.decode_tps > 0.0 {
            self.decode_tps
        } else {
            1.0
        }
    }

    /// Seconds of compute per token for holding `layers` of `num_layers_total`.
    ///
    /// `decode_tps` covers the whole model, so a stage holding a `share` of
    /// the layers costs `share / decode_tps` seconds per token.
    pub fn compute_secs(&self, layers: u32, num_layers_total: u32) -> f64 {
        (layers as f64 / num_layers_total.max(1) as f64) / self.sane_tps()
    }

    /// Seconds to move one token's activation to the next stage.
    pub fn comm_secs(&self) -> f64 {
        let bytes_per_sec = self.bandwidth_mbps * 1_000_000.0 / 8.0;
        if !bytes_per_sec.is_finite() || bytes_per_sec <= 0.0 {
            return 0.0;
        }
        ACTIVATION_BYTES_PER_TOKEN as f64 / bytes_per_sec
    }

    /// Per-token stage time: compute plus one activation hop. Every stage
    /// pays one hop (the tail's hop delivers the token/commit downstream to
    /// the coordinator), so identical devices split purely by layer count.
    pub fn stage_time_secs(&self, layers: u32, num_layers_total: u32) -> f64 {
        self.compute_secs(layers, num_layers_total) + self.comm_secs()
    }
}

/// Stable plan id for a membership: FNV-1a over layer count + node ids.
fn plan_id_for(num_layers: u32, devices: &[DeviceSpec]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    for b in num_layers.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    for d in devices {
        for b in d.node_id.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// Number of contiguous compositions: C(`layers`-1, `stages`-1).
fn composition_count(layers: u32, stages: usize) -> u128 {
    if stages == 0 || layers == 0 {
        return 0;
    }
    let n = layers as u128 - 1;
    let mut k = stages as u128 - 1;
    if k > n {
        return 0;
    }
    k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc.saturating_mul(n - i) / (i + 1);
    }
    acc
}

/// Fallback for huge search spaces: derive per-device [`Calibration`] from
/// the share model and reuse [`greedy_partition`].
fn proportional_fallback(num_layers: u32, devices: &[DeviceSpec]) -> Vec<StageRange> {
    let cals: Vec<Calibration> = devices
        .iter()
        .map(|d| {
            let tps = if d.decode_tps.is_finite() && d.decode_tps > 0.0 {
                d.decode_tps
            } else {
                1.0
            };
            Calibration {
                ms_per_layer_decode: 1000.0 / (tps * num_layers.max(1) as f64),
                ms_per_layer_prefill_chunk: 0.0,
                hop_ms: d.comm_secs() * 1000.0,
            }
        })
        .collect();
    greedy_partition(num_layers, &cals)
}

/// Assign contiguous layer ranges to `devices`, minimizing the max per-stage
/// time (compute + one activation hop per stage).
///
/// Exhaustively searches contiguous partitions (28 layers / <=5 stages is at
/// most ~18k compositions); ties break toward balanced splits, then toward
/// edge-symmetric ones (so 3 identical devices yield 9/10/9, matching the
/// catalog `default_plan`), then lexicographically for determinism.
pub fn plan_layers(num_layers: u32, devices: &[DeviceSpec]) -> PipelinePlan {
    assert!(!devices.is_empty(), "need at least one device");
    assert!(num_layers >= 1, "need at least one layer");
    assert!(
        num_layers >= devices.len() as u32,
        "more stages than layers"
    );

    let n = devices.len();
    if n == 1 {
        return PipelinePlan {
            plan_id: plan_id_for(num_layers, devices),
            stages: vec![StageRange::new(0, num_layers - 1)],
        };
    }
    if composition_count(num_layers, n) > 5_000_000 {
        return PipelinePlan {
            plan_id: plan_id_for(num_layers, devices),
            stages: proportional_fallback(num_layers, devices),
        };
    }

    // Per-stage cost table: cost[stage][k] for holding k layers.
    let mut cost = vec![vec![0.0f64; num_layers as usize + 1]; n];
    for (s, d) in devices.iter().enumerate() {
        for k in 1..=num_layers {
            cost[s][k as usize] = d.stage_time_secs(k, num_layers);
        }
    }

    // (max_time, layer-count range, |first - last|, counts).
    let mut best: Option<(f64, u32, u32, Vec<u32>)> = None;    let mut counts = vec![1u32; n];

    fn rec(
        stage: usize,
        rest: u32,
        counts: &mut Vec<u32>,
        cost: &[Vec<f64>],
        best: &mut Option<(f64, u32, u32, Vec<u32>)>,
    ) {
        const EPS: f64 = 1e-9;
        let n = counts.len();
        if stage + 1 == n {
            counts[stage] = rest;
            let mut max_t = 0.0f64;
            let (mut lo, mut hi) = (u32::MAX, 0u32);
            for (s, &c) in counts.iter().enumerate() {
                let t = cost[s][c as usize];
                if t > max_t {
                    max_t = t;
                }
                lo = lo.min(c);
                hi = hi.max(c);
            }
            let range = hi - lo;
            let asym = counts[0].abs_diff(counts[n - 1]);
            let better = match &*best {
                None => true,
                Some((bt, br, ba, bc)) => {
                    if max_t < bt - EPS {
                        true
                    } else if (max_t - bt).abs() <= EPS * bt.abs().max(1.0) {
                        if range != *br {
                            range < *br
                        } else if asym != *ba {
                            asym < *ba
                        } else {
                            counts.as_slice() < bc.as_slice()
                        }
                    } else {
                        false
                    }
                }
            };
            if better {
                *best = Some((max_t, range, asym, counts.clone()));
            }
            return;
        }
        let stages_after = (n - stage - 1) as u32;
        let max_k = rest - stages_after; // leave >= 1 layer per remaining stage
        for k in 1..=max_k {
            counts[stage] = k;
            rec(stage + 1, rest - k, counts, cost, best);
        }
    }

    rec(0, num_layers, &mut counts, &cost, &mut best);
    let (_, _, _, counts) = best.expect("at least one partition exists");

    let mut stages = Vec::with_capacity(n);
    let mut start = 0u32;
    for c in counts {
        let end = start + c - 1;
        stages.push(StageRange::new(start, end));
        start = end + 1;
    }
    PipelinePlan {
        plan_id: plan_id_for(num_layers, devices),
        stages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identical_devices(n: usize) -> Vec<DeviceSpec> {
        (0..n)
            .map(|i| {
                DeviceSpec::new(
                    format!("node-{i}"),
                    16.6,   // decode_tps reference (docs/bench-baseline.json)
                    1000.0, // bandwidth_mbps
                    1024,   // kv_budget_mib
                )
            })
            .collect()
    }

    #[test]
    fn single_device_takes_all_layers() {
        let devices = identical_devices(1);
        let plan = plan_layers(TOTAL_LAYERS, &devices);
        assert_eq!(plan.stages, vec![StageRange::new(0, TOTAL_LAYERS - 1)]);
        assert_eq!(plan.num_layers(), TOTAL_LAYERS);
    }

    #[test]
    fn three_identical_devices_split_9_10_9() {
        let devices = identical_devices(3);
        let plan = plan_layers(TOTAL_LAYERS, &devices);
        assert_eq!(plan.stages, PipelinePlan::balanced_9_10_9().stages);
        assert_eq!(
            plan.stages,
            vec![
                StageRange::new(0, 8),
                StageRange::new(9, 18),
                StageRange::new(19, 27),
            ]
        );
    }

    #[test]
    fn faster_device_gets_more_layers() {
        let devices = vec![
            DeviceSpec::new("fast", 33.2, 1000.0, 1024),
            DeviceSpec::new("slow", 16.6, 1000.0, 1024),
        ];
        let plan = plan_layers(TOTAL_LAYERS, &devices);
        assert_eq!(plan.num_layers(), TOTAL_LAYERS);
        assert_eq!(plan.stages.len(), 2);
        // 2:1 throughput ratio settles at 19/9 (equalized stage times).
        assert_eq!(plan.stages[0].len(), 19);
        assert_eq!(plan.stages[1].len(), 9);
        assert_eq!(plan.stages[0], StageRange::new(0, 18));
        assert_eq!(plan.stages[1], StageRange::new(19, 27));
    }

    #[test]
    fn plan_is_contiguous_and_covers_all_layers() {
        let devices = identical_devices(3);
        let plan = plan_layers(TOTAL_LAYERS, &devices);
        assert_eq!(plan.stages[0].start, 0);
        assert_eq!(plan.stages[plan.stages.len() - 1].end, TOTAL_LAYERS - 1);
        for w in plan.stages.windows(2) {
            assert_eq!(w[0].end + 1, w[1].start);
        }
    }
}
