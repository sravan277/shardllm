//! Parsing `dllm_shim_session_report`'s JSON into real Rust types.
//!
//! The shim reports what llama.cpp *actually* did, not what the coordinator
//! asked for. The distinction matters enough to be worth its own module:
//!
//! - `tensor_split` is the request, echoed back verbatim.
//! - `layer_owner[i]` is the ggml device index holding layer `i`'s tensors,
//!   read out of `llama_model::dev_layer()`. llama.cpp falls back to another
//!   device when a tensor does not fit where the ratio asked for it to go, so
//!   this routinely differs from the request.
//!
//! When the two disagree the coordinator shows **both** (ADR-032); it never
//! reconciles them into a single tidy number, because a reconciled number is a
//! number nobody measured.
//!
//! `layer_owner` is `Option`, never defaulted: the shim emits an explicit
//! `null` (with a `layer_owner_note` explaining why) when the internal header
//! stops exposing `dev_layer()`. Guessing from `tensor_split` in that case
//! would be precisely the "restate intent instead of reporting reality" failure
//! the shim's own comments refuse.

use serde::{Deserialize, Serialize};

/// Timing snapshot from the most recent `generate()` call.
///
/// All fields are `f64`/`i32` because the shim formats them with
/// `std::to_string`, which emits `0.000000` / `3.500000` — valid JSON numbers.
/// A token count of 0 with `decode_ms == 0` is the honest "nothing generated"
/// shape, not a missing measurement, so it is kept as a real `0.0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShimTiming {
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub predicted_per_second: f64,
    pub n_prompt_tokens: i32,
    pub n_generated: i32,
}

impl ShimTiming {
    /// Measured decode cost of one generated token, or `None` when it cannot be
    /// computed (nothing generated, or a zero/negative timer). Never `0.0` as a
    /// stand-in for "unknown".
    pub fn decode_ms_per_token(&self) -> Option<f64> {
        if self.n_generated > 0 && self.decode_ms > 0.0 {
            Some(self.decode_ms / self.n_generated as f64)
        } else {
            None
        }
    }
}

/// Everything `dllm_shim_session_report` produced.
///
/// `Option` fields are `Option` because the shim emits JSON `null` for them,
/// and this codebase's rule is that a null means "not measured" — never fill
/// one in with a guess.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelReport {
    /// Transformer layers in the loaded model (29 in the proven run).
    pub n_layer: i32,
    /// Context size llama.cpp actually gave the session.
    pub n_ctx: i32,
    /// Layers **not** on the CPU device, counted from `dev_layer()`.
    pub n_gpu_layers: i32,
    /// The `tensor_split` the caller requested, echoed verbatim. `None` when
    /// the session was opened without one (everything local).
    #[serde(default)]
    pub tensor_split: Option<Vec<f32>>,
    /// `ggml_backend_dev_count()` at report time: local devices plus every
    /// registered RPC device.
    pub devices: i32,
    /// `None` when the shim omitted the block entirely.
    #[serde(default)]
    pub timing: Option<ShimTiming>,
    /// ggml device index per layer. `None` = the shim could not read
    /// `dev_layer()` back and said so in `layer_owner_note`.
    #[serde(default)]
    pub layer_owner: Option<Vec<i32>>,
    /// Why `layer_owner` is null, when it is.
    #[serde(default)]
    pub layer_owner_note: Option<String>,
}

impl ModelReport {
    /// Distinct ggml device indices that hold at least one layer.
    ///
    /// The end-to-end assertion lives here: a distributed session must report
    /// more than one, and `"layers actually moved"` is exactly this number
    /// being greater than 1.
    pub fn distinct_layer_owners(&self) -> Vec<i32> {
        let mut seen: Vec<i32> = Vec::new();
        for &dev in self.layer_owner.iter().flatten() {
            if !seen.contains(&dev) {
                seen.push(dev);
            }
        }
        seen.sort_unstable();
        seen
    }

    /// Per-ggml-device layer counts, ascending by device index. Measured.
    pub fn layers_per_device(&self) -> Vec<(i32, u32)> {
        let mut counts: Vec<(i32, u32)> = Vec::new();
        for &dev in self.layer_owner.iter().flatten() {
            match counts.iter_mut().find(|(d, _)| *d == dev) {
                Some(slot) => slot.1 += 1,
                None => counts.push((dev, 1)),
            }
        }
        counts.sort_by_key(|(d, _)| *d);
        counts
    }

    /// Number of layers llama.cpp put on a non-local device, as measured from
    /// `layer_owner`. Differs from `n_gpu_layers` only when `layer_owner` is
    /// unavailable, in which case this returns `None`.
    pub fn measured_offloaded_layers(&self) -> Option<u32> {
        let owner = self.layer_owner.as_ref()?;
        // ggml slot 0 is the first local device on every supported build (the
        // shim filters ACCEL backends and the registry constructor registers
        // CPU first). Anything above 0 is remote or another local accelerator;
        // the honest split is therefore reported per-slot and interpreted by
        // the caller that knows which slots it registered.
        Some(owner.iter().filter(|&&d| d > 0).count() as u32)
    }
}

/// Why a report could not be read.
#[derive(Debug, thiserror::Error)]
pub enum ReportError {
    #[error("dllm_shim_session_report failed: {0}")]
    Shim(String),
    /// The buffer was too small and the shim truncated the JSON.
    #[error("report JSON was truncated at {len} bytes; raise REPORT_BUFFER_BYTES")]
    Truncated { len: usize },
    #[error("report JSON is not valid: {0}")]
    Json(#[from] serde_json::Error),
}

/// Bytes the shim is given for `dllm_shim_session_report`.
///
/// 28 layers of `layer_owner` plus the timing block is a few hundred bytes, so
/// 8 KiB is generous; the shim truncates rather than overflowing, and
/// [`parse_report`] detects truncation instead of returning half a JSON object.
pub const REPORT_BUFFER_BYTES: usize = 8 * 1024;

/// Parse a report out of the NUL-terminated buffer the shim filled.
///
/// Takes the raw bytes rather than a `&str` because the shim writes a C string:
/// the caller knows the length it passed in, and slicing at the first NUL
/// mirrors what C++ would have seen. A buffer with **no** NUL is reported as
/// [`ReportError::Truncated`] — the shim's `copy_bounded` always terminates
/// inside the length it was given, so a missing terminator means the JSON was
/// clipped mid-value and half a report is worse than none.
pub fn parse_report(buf: &[u8]) -> Result<ModelReport, ReportError> {
    let end = buf.iter().position(|&b| b == 0).ok_or(ReportError::Truncated {
        len: buf.len(),
    })?;
    let text = std::str::from_utf8(&buf[..end])
        .map_err(|e| ReportError::Json(serde_json::Error::io(std::io::Error::other(e))))?;
    Ok(serde_json::from_str(text)?)
}

/// [`parse_report`] for a string literal.
///
/// Appends the NUL the C side would have written, which is the shape every
/// real caller sees. Tests use this so they read as JSON rather than as buffer
/// plumbing.
pub fn parse_report_str(json: &str) -> Result<ModelReport, ReportError> {
    let mut buf = Vec::with_capacity(json.len() + 1);
    buf.extend_from_slice(json.as_bytes());
    buf.push(0);
    parse_report(&buf)
}

/// Render a `tensor_split` for logs and API output: one entry per ggml device
/// slot, trailing zeros dropped (llama.cpp's buffer is `llama_max_devices()`
/// long and the tail is always zero, so printing it all is noise).
pub fn format_tensor_split(weights: &[f32]) -> String {
    let last = weights
        .iter()
        .rposition(|w| *w != 0.0)
        .map(|i| i + 1)
        .unwrap_or(0);
    let shown: Vec<String> = weights[..last]
        .iter()
        .map(|w| {
            if w.fract() == 0.0 {
                format!("{w:.0}")
            } else {
                format!("{w:.3}")
            }
        })
        .collect();
    format!("[{}]", shown.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact shape the proven C++ run produced, lightly trimmed.
    const REAL_REPORT: &str = r#"{"n_layer":29,"n_ctx":2048,"n_gpu_layers":14,"tensor_split":[15,14],"devices":2,"timing":{"prefill_ms":112.500000,"decode_ms":1843.250000,"predicted_per_second":13.017047,"n_prompt_tokens":6,"n_generated":24},"layer_owner":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]}"#;

    #[test]
    fn parses_the_proven_report_shape() {
        let r = parse_report_str(REAL_REPORT).unwrap();
        assert_eq!(r.n_layer, 29);
        assert_eq!(r.n_ctx, 2048);
        assert_eq!(r.n_gpu_layers, 14);
        assert_eq!(r.tensor_split, Some(vec![15.0, 14.0]));
        assert_eq!(r.devices, 2);
        let t = r.timing.expect("timing block");
        assert_eq!(t.n_prompt_tokens, 6);
        assert_eq!(t.n_generated, 24);
        assert!((t.decode_ms - 1843.25).abs() < 1e-9);
        // 1843.25 / 24
        let per = t.decode_ms_per_token().expect("measurable");
        assert!((per - 76.80).abs() < 0.01, "got {per}");
        assert_eq!(r.layer_owner.as_ref().unwrap().len(), 29);
        assert_eq!(r.distinct_layer_owners(), vec![0, 1]);
        assert_eq!(r.layers_per_device(), vec![(0, 15), (1, 14)]);
        assert_eq!(r.measured_offloaded_layers(), Some(14));
    }

    #[test]
    fn a_real_single_device_report_reports_one_owner() {
        // What `dllm serve` produces with no workers paired: honest, and NOT a
        // failure — the distribution just is not distributed.
        let js = r#"{"n_layer":28,"n_ctx":4096,"n_gpu_layers":0,"tensor_split":null,"devices":1,
                    "timing":{"prefill_ms":10.0,"decode_ms":100.0,"predicted_per_second":25.0,
                              "n_prompt_tokens":8,"n_generated":25},"layer_owner":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}"#;
        let r = parse_report_str(js).unwrap();
        assert_eq!(r.tensor_split, None);
        assert_eq!(r.distinct_layer_owners(), vec![0]);
        assert_eq!(r.layers_per_device(), vec![(0, 28)]);
        assert_eq!(r.measured_offloaded_layers(), Some(0));
        assert_eq!(r.n_gpu_layers, 0);
    }

    #[test]
    fn layer_owner_null_is_preserved_as_not_measured() {
        let js = r#"{"n_layer":28,"n_ctx":4096,"n_gpu_layers":28,"tensor_split":[14,14],"devices":2,
                    "timing":{"prefill_ms":1.0,"decode_ms":2.0,"predicted_per_second":1.0,
                              "n_prompt_tokens":1,"n_generated":2},
                    "layer_owner":null,
                    "layer_owner_note":"dev_layer() unavailable in this build"}"#;
        let r = parse_report_str(js).unwrap();
        assert_eq!(r.layer_owner, None);
        assert!(
            r.layer_owner_note
                .as_deref()
                .unwrap_or_default()
                .contains("dev_layer")
        );
        // No fallback to the requested split: that would be intent, not fact.
        assert_eq!(r.distinct_layer_owners(), Vec::<i32>::new());
        assert_eq!(r.layers_per_device(), Vec::new());
        assert_eq!(r.measured_offloaded_layers(), None);
    }

    #[test]
    fn decode_ms_per_token_is_none_when_nothing_was_generated() {
        let t = ShimTiming {
            prefill_ms: 0.0,
            decode_ms: 0.0,
            predicted_per_second: 0.0,
            n_prompt_tokens: 9,
            n_generated: 0,
        };
        assert_eq!(t.decode_ms_per_token(), None);
        let t2 = ShimTiming {
            decode_ms: 50.0,
            n_generated: 0,
            ..t
        };
        assert_eq!(t2.decode_ms_per_token(), None);
        assert_eq!(
            ShimTiming {
                decode_ms: 50.0,
                n_generated: 5,
                ..t2
            }
            .decode_ms_per_token(),
            Some(10.0)
        );
    }

    #[test]
    fn truncated_buffer_is_an_error_not_a_half_report() {
        let buf = REPORT_BUFFER_BYTES; // all 0x7b = '{', i.e. no NUL anywhere
        let mut full = vec![b'{'; buf];
        full[0] = b'{';
        assert!(matches!(
            parse_report(&full),
            Err(ReportError::Truncated { len }) if len == buf
        ));
    }

    #[test]
    fn nul_terminated_buffer_is_sliced_like_a_c_string() {
        let mut buf = REAL_REPORT.as_bytes().to_vec();
        buf.push(0);
        buf.extend_from_slice(b"garbage after the NUL");
        let r = parse_report(&buf).unwrap();
        assert_eq!(r.n_layer, 29);
    }

    #[test]
    fn malformed_json_names_the_parse_failure() {
        let err = parse_report_str("{\"n_layer\":").unwrap_err();
        assert!(matches!(err, ReportError::Json(_)));
        assert!(err.to_string().contains("not valid"));
    }

    #[test]
    fn tensor_split_formatting_drops_the_always_zero_tail() {
        assert_eq!(format_tensor_split(&[15.0, 14.0, 0.0, 0.0]), "[15, 14]");
        assert_eq!(format_tensor_split(&[9.0, 10.0, 9.0]), "[9, 10, 9]");
        assert_eq!(format_tensor_split(&[4.3333333]), "[4.333]");
        assert_eq!(format_tensor_split(&[0.0, 0.0]), "[]");
        assert_eq!(format_tensor_split(&[]), "[]");
    }
}
