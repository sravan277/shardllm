//! Turning a pipeline plan into a llama.cpp `tensor_split` array.
//!
//! This is the whole point of the crate boundary: `crates/dllm-core` computes
//! a real [`PipelinePlan`](https://docs.rs/dllm-core) (contiguous
//! `StageRange`s chosen to minimise the slowest stage), and llama.cpp wants a
//! flat `float*` of *relative* weights indexed by ggml device slot. Nothing
//! else in the codebase bridges those two, so before this module the plan was
//! computed and then thrown away at render time.
//!
//! # The ordering invariant (read this before changing anything)
//!
//! `dllm_shim_session_open` hands llama.cpp a **NULL-terminated
//! `ggml_backend_dev_t*` array consumed verbatim, in order** — local devices
//! first, then RPC devices in *registration* order. `tensor_split` is indexed
//! by position in that very array. So:
//!
//! ```text
//! plan stage 0  <->  ggml device slot 0            (local)
//! plan stage 1  <->  the slots of the 1st endpoint registered
//! plan stage 2  <->  the slots of the 2nd endpoint registered
//! ...
//! ```
//!
//! Two rules fall out of that and both are enforced by the caller
//! ([`crate::endpoint::RpcRegistry::register_in_order`]):
//! 1. **registration order must equal stage order.** If stage 2's worker is
//!    registered before stage 1's, then `tensor_split[1]` lands on stage 2's
//!    worker and every reported `layer_owner` index silently means the wrong
//!    device. There is no way to detect this from inside the shim.
//! 2. **an endpoint must be registered at most once.**
//!    `dllm_shim_add_rpc_server` appends to a process-global device list and
//!    deliberately does *not* de-duplicate, so a double-add silently doubles
//!    the device slots and shifts every later stage.
//!
//! # Why weights, not fractions
//!
//! The header is explicit that `tensor_split` holds *relative* weights that
//! llama.cpp normalises (`{1,1}` is an even split, `{2,1}` is two-to-one).
//! Feeding it raw layer counts is therefore both correct and the most honest
//! option available: the number in the report is the number of layers the
//! planner asked for, not a rounded percentage of anything.
//!
//! Everything here is pure arithmetic — no FFI — so the whole module is
//! unit-testable on a machine that has never built `dllm_shim.dll`.

use std::fmt;

/// A stage's placement, resolved against the *devices that actually exist*.
///
/// `first_slot` is the ggml device index the stage's first device occupies and
/// `devices` is how many consecutive slots it owns. A worker exposing two ggml
/// devices owns two slots, and its layer count is shared between them, which
/// keeps `tensor_split` aligned with the device array no matter how many
/// devices a worker happens to expose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageSlot {
    /// Index into the plan's stage list (kept so a caller can attribute a slot
    /// back to a `device_id`).
    pub stage: usize,
    /// Layers the plan assigned to that stage.
    pub layers: u32,
    /// First ggml device slot owned by this stage.
    pub first_slot: usize,
    /// How many consecutive ggml device slots it owns (>= 1).
    pub devices: usize,
}

/// Why a `tensor_split` could not be derived.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitError {
    /// The plan has no stages at all; there is nothing to distribute.
    Empty,
    /// A stage owns zero ggml devices, which llama.cpp cannot express.
    ZeroDeviceStage { stage: usize },
    /// A stage was assigned zero layers. Legal to *hold* (the planner never
    /// emits it) but the caller almost certainly has a mismatch, so it is a
    /// loud error rather than a silent `0.0` weight.
    ZeroLayerStage { stage: usize },
}

impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "plan has no stages; nothing to distribute"),
            Self::ZeroDeviceStage { stage } => {
                write!(f, "stage {stage} owns zero ggml devices")
            }
            Self::ZeroLayerStage { stage } => write!(
                f,
                "stage {stage} was assigned 0 layers; the plan and the device list disagree"
            ),
        }
    }
}

impl std::error::Error for SplitError {}

/// The resolved layout: one [`StageSlot`] per stage plus the `tensor_split`
/// weights, in ggml device order.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceLayout {
    pub slots: Vec<StageSlot>,
    /// `tensor_split` weights, one per ggml device slot, in slot order.
    pub weights: Vec<f32>,
}

/// Derive `tensor_split` weights from per-stage layer counts.
///
/// `stage_layers[i]` is how many transformer layers plan stage `i` holds;
/// `devices_per_stage[i]` is how many consecutive ggml device slots that stage
/// owns (1 for a plain local CPU or a single-device worker, >1 when a worker
/// exposes several ggml backends). The two vectors must be the same length and
/// must be in **plan stage order** — see the module docs for why order is
/// load-bearing.
///
/// A stage's layer count is divided evenly across the slots it owns, so the
/// weights always sum back to the total layer count:
///
/// ```
/// use dllm_shim::split::tensor_split_from_stages;
///
/// // 9/10/9 over one local CPU + two single-device workers.
/// let w = tensor_split_from_stages(&[9, 10, 9], &[1, 1, 1]).unwrap();
/// assert_eq!(w.weights, vec![9.0, 10.0, 9.0]);
/// assert_eq!(w.weights.iter().map(|x| *x as u32).sum::<u32>(), 28);
///
/// // A worker that exposes two ggml devices keeps its slot block aligned:
/// // stage 1's 10 layers become 5 + 5 across slots 1 and 2.
/// let w = tensor_split_from_stages(&[9, 10, 9], &[1, 2, 1]).unwrap();
/// assert_eq!(w.weights, vec![9.0, 5.0, 5.0, 9.0]);
/// assert_eq!(w.slots[1].first_slot, 1);
/// assert_eq!(w.slots[1].devices, 2);
/// assert_eq!(w.slots[2].first_slot, 3);
/// ```
pub fn tensor_split_from_stages(
    stage_layers: &[u32],
    devices_per_stage: &[usize],
) -> Result<DeviceLayout, SplitError> {
    if stage_layers.is_empty() {
        return Err(SplitError::Empty);
    }
    if stage_layers.len() != devices_per_stage.len() {
        // Not a `SplitError` variant of its own on purpose: it is a caller bug
        // (plan arity vs registered device arity), and the message below names
        // both numbers so the mismatch is obvious in a log line.
        return Err(SplitError::ZeroDeviceStage {
            stage: stage_layers.len().min(devices_per_stage.len()),
        });
    }

    let mut slots = Vec::with_capacity(stage_layers.len());
    let mut weights: Vec<f32> = Vec::with_capacity(devices_per_stage.len());
    let mut next_slot = 0usize;

    for (stage, (&layers, &devices)) in stage_layers
        .iter()
        .zip(devices_per_stage.iter())
        .enumerate()
    {
        if devices == 0 {
            return Err(SplitError::ZeroDeviceStage { stage });
        }
        if layers == 0 {
            return Err(SplitError::ZeroLayerStage { stage });
        }
        slots.push(StageSlot {
            stage,
            layers,
            first_slot: next_slot,
            devices,
        });
        // Relative weights, normalised by llama.cpp. Integer layers divided by
        // an integer device count is exact for the common case (1 device), and
        // f32 has enough mantissa for `layers <= ~16.7M` either way.
        let per_device = layers as f32 / devices as f32;
        for _ in 0..devices {
            weights.push(per_device);
        }
        next_slot += devices;
    }

    Ok(DeviceLayout { slots, weights })
}

/// Layers each stage actually holds, given a `tensor_split`-shaped weight list
/// and the total layer count of the model.
///
/// Inverse of [`tensor_split_from_stages`] in the single-device-per-stage case,
/// and used to explain a *requested* plan against a model whose real layer
/// count differs from the planner's `TOTAL_LAYERS`. Kept separate from
/// `layer_owner` on purpose: this is still intent, not measurement.
///
/// Rounding is **largest-remainder**, so the parts always sum to `n_layer`
/// exactly. Independent `round()` calls would not: `[1, 1]` over 3 layers
/// rounds to `[2, 2]` = 4 layers, which is how a plan ends up asking for more
/// layers than the model has.
///
/// ```
/// use dllm_shim::split::layers_from_tensor_split;
/// assert_eq!(layers_from_tensor_split(&[9.0, 10.0, 9.0], 28), vec![9, 10, 9]);
/// assert_eq!(layers_from_tensor_split(&[1.0, 1.0], 3), vec![2, 1]);
/// assert_eq!(layers_from_tensor_split(&[0.0, 0.0], 28), vec![0, 0]);
/// ```
pub fn layers_from_tensor_split(weights: &[f32], n_layer: u32) -> Vec<u32> {
    let total: f64 = weights.iter().map(|w| *w as f64).sum();
    if total <= 0.0 || n_layer == 0 {
        return vec![0; weights.len()];
    }
    let exact: Vec<f64> = weights
        .iter()
        .map(|w| (*w as f64) * (n_layer as f64) / total)
        .collect();
    let mut out: Vec<u32> = exact.iter().map(|e| e.floor().max(0.0) as u32).collect();
    let mut assigned: u32 = out.iter().sum();
    // Hand the leftover layers to the largest fractional parts first, ties by
    // slot order so the result is deterministic.
    let mut order: Vec<usize> = (0..exact.len()).collect();
    order.sort_by(|&a, &b| {
        let fa = exact[a] - exact[a].floor();
        let fb = exact[b] - exact[b].floor();
        fb.partial_cmp(&fa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut i = 0usize;
    while assigned < n_layer && !order.is_empty() {
        out[order[i % order.len()]] += 1;
        assigned += 1;
        i += 1;
    }
    out
}

/// Collapse a flat `layer_owner` array (one ggml device index per transformer
/// layer) into contiguous per-device runs.
///
/// This is the *measured* counterpart to [`tensor_split_from_stages`]: the
/// shim reads it back out of `llama_model::dev_layer()`, i.e. out of the
/// assignment llama.cpp really made, so it disagrees with the request whenever
/// llama.cpp had to fall back (e.g. a tensor did not fit on the requested
/// device). Grouping only *runs* means a device that somehow ended up with two
/// disjoint ranges produces two entries rather than a silently wrong
/// `layer_start`/`layer_end` pair.
///
/// ```
/// use dllm_shim::split::contiguous_runs;
/// assert_eq!(
///     contiguous_runs(&[0, 0, 0, 1, 1]),
///     vec![(0, 0usize, 3usize), (1, 3usize, 2usize)]
/// );
/// // Non-contiguous ownership is reported as two honest runs, not merged.
/// assert_eq!(
///     contiguous_runs(&[1, 0, 1]),
///     vec![(1, 0usize, 1usize), (0, 1usize, 1usize), (1, 2usize, 1usize)]
/// );
/// ```
pub fn contiguous_runs(layer_owner: &[i32]) -> Vec<(i32, usize, usize)> {
    let mut out: Vec<(i32, usize, usize)> = Vec::new();
    for (layer, &dev) in layer_owner.iter().enumerate() {
        match out.last_mut() {
            Some(last) if last.0 == dev && last.1 + last.2 == layer => last.2 += 1,
            _ => out.push((dev, layer, 1)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_of_n_stages_produces_n_weights_that_sum_to_the_layer_count() {
        for counts in [
            vec![28u32],
            vec![15, 13],
            vec![9, 10, 9],
            vec![8, 7, 7, 6],
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ] {
            let devices: Vec<usize> = counts.iter().map(|_| 1).collect();
            let layout = tensor_split_from_stages(&counts, &devices).expect("valid plan");
            assert_eq!(layout.weights.len(), counts.len(), "one weight per stage");
            let sum: f32 = layout.weights.iter().sum();
            assert_eq!(sum, counts.iter().sum::<u32>() as f32);
            // Slots are dense and in stage order — the invariant the shim's
            // verbatim device array depends on.
            let mut expect = 0usize;
            for s in &layout.slots {
                assert_eq!(s.first_slot, expect);
                expect += s.devices;
            }
            assert_eq!(expect, layout.weights.len());
        }
    }

    #[test]
    fn changing_stage_layer_counts_changes_the_weights() {
        let a = tensor_split_from_stages(&[15, 13], &[1, 1]).unwrap();
        let b = tensor_split_from_stages(&[19, 9], &[1, 1]).unwrap();
        assert_eq!(a.weights, vec![15.0, 13.0]);
        assert_eq!(b.weights, vec![19.0, 9.0]);
        assert_ne!(a.weights, b.weights);
    }

    #[test]
    fn multi_device_stage_splits_its_weight_but_keeps_its_slot_block() {
        let layout = tensor_split_from_stages(&[15, 13], &[1, 3]).unwrap();
        // 13 layers over 3 devices -> 4.333 each; the sum is still 13.
        assert_eq!(layout.weights.len(), 4);
        assert_eq!(layout.weights[0], 15.0);
        assert!(layout.weights[1..].iter().all(|w| (*w - 13.0 / 3.0).abs() < 1e-5));
        let total: f32 = layout.weights.iter().sum();
        assert!((total - 28.0).abs() < 1e-3, "got {total}");
        assert_eq!(layout.slots[1].first_slot, 1);
        assert_eq!(layout.slots[1].devices, 3);
    }

    #[test]
    fn arity_mismatch_and_zero_devices_are_errors_not_silent_zeroes() {
        assert_eq!(tensor_split_from_stages(&[], &[]), Err(SplitError::Empty));
        assert_eq!(
            tensor_split_from_stages(&[14, 14], &[1]),
            Err(SplitError::ZeroDeviceStage { stage: 1 })
        );
        assert_eq!(
            tensor_split_from_stages(&[14, 0, 14], &[1, 1, 1]),
            Err(SplitError::ZeroLayerStage { stage: 1 })
        );
        assert_eq!(
            tensor_split_from_stages(&[14, 14], &[1, 0]),
            Err(SplitError::ZeroDeviceStage { stage: 1 })
        );
    }

    #[test]
    fn layers_from_tensor_split_inverts_the_weights_and_still_covers_the_model() {
        assert_eq!(layers_from_tensor_split(&[9.0, 10.0, 9.0], 28), vec![9, 10, 9]);
        // Ragged weights round, and the parts still sum to the model size -
        // independent `round()` would give [2, 2] = 4 layers over a 3-layer model.
        assert_eq!(layers_from_tensor_split(&[1.0, 1.0], 3), vec![2, 1]);
        let got = layers_from_tensor_split(&[15.0, 14.0], 29);
        assert_eq!(got.iter().sum::<u32>(), 29);
        // Degenerate input must not divide by zero or invent layers.
        assert_eq!(layers_from_tensor_split(&[0.0, 0.0], 28), vec![0, 0]);
        assert_eq!(layers_from_tensor_split(&[9.0, 10.0], 0), vec![0, 0]);
    }

    #[test]
    fn contiguous_runs_groups_by_device_and_position() {
        assert_eq!(contiguous_runs(&[0, 0, 0, 1, 1]), vec![(0, 0, 3), (1, 3, 2)]);
        assert_eq!(contiguous_runs(&[]), Vec::<(i32, usize, usize)>::new());
        assert_eq!(contiguous_runs(&[2]), vec![(2, 0, 1)]);
    }
}
