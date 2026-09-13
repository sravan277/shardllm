//! dllm-net: QUIC framing primitives (ALPN, activation frames, ACKs).
//!
//! Pure Rust only. QUIC transport wiring (quinn + rustls) builds on these
//! types in Phase 3; this crate only owns the wire format + dev identity.

/// ALPN protocol id — must match on both ends or the handshake fails silently.
pub const ALPN: &[u8] = b"dllm/1";

/// Wire magic prefix for every activation frame.
pub const FRAME_MAGIC: &[u8; 5] = b"DLLM1";

/// Hard cap for a decoded frame (8 MiB). Slow workers must exert backpressure
/// instead of buffering unbounded activations.
pub const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// Tensor dtype/layout tag carried in [`ActivationFrame::tensor_format`].
///
/// Values match `contracts/activation-frame.md`: `0`=fp16, `1`=fp32, `2`=q8.
pub mod tensor_format {
    pub const F16: u8 = 0;
    pub const F32: u8 = 1;
    pub const Q8: u8 = 2;
    pub const U8_BYTES: u8 = 255;
}

/// Activation tensor forwarded from one pipeline stage to the next.
///
/// Field names mirror MASTER_PLAN §8 (`session_id`, `pipeline_plan_id`,
/// `token_position`, `source_stage`, `target_stage`, `tensor_format`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ActivationFrame {
    pub session_id: u128,
    pub plan_id: u64,
    pub token_position: u32,
    pub source_stage: u8,
    pub target_stage: u8,
    pub tensor_format: u8,
    pub payload: Vec<u8>,
}

/// Inference-level ACK vocabulary (transport ACK != inference ACK).
///
/// Optimistic protocol (research 00-index delta): stages append
/// `KvTentative` without waiting; the coordinator piggybacks
/// `Committed(pos-1)` on the next dispatch; abort = `Truncate(pos)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Ack {
    Received { pos: u32, token: u32 },
    Computed { pos: u32, token: u32 },
    KvTentative { pos: u32, token: u32 },
    Committed { pos: u32, token: u32 },
    Truncate { pos: u32, token: u32 },
}

impl Ack {
    pub fn pos(self) -> u32 {
        match self {
            Ack::Received { pos, .. }
            | Ack::Computed { pos, .. }
            | Ack::KvTentative { pos, .. }
            | Ack::Committed { pos, .. }
            | Ack::Truncate { pos, .. } => pos,
        }
    }

    pub fn token(self) -> u32 {
        match self {
            Ack::Received { token, .. }
            | Ack::Computed { token, .. }
            | Ack::KvTentative { token, .. }
            | Ack::Committed { token, .. }
            | Ack::Truncate { token, .. } => token,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NetError {
    #[error("frame exceeds 8 MiB cap ({0} bytes)")]
    TooLarge(usize),
    #[error("bad frame magic")]
    BadMagic,
    #[error("length prefix mismatch: declared {declared}, actual {actual}")]
    LengthMismatch { declared: usize, actual: usize },
    #[error("postcard: {0}")]
    Postcard(#[from] postcard::Error),
    #[error("rcgen: {0}")]
    Rcgen(#[from] rcgen::Error),
}

/// Encode a frame as `MAGIC(5) || u32-LE len || postcard bytes`.
pub fn encode_frame(frame: &ActivationFrame) -> Result<Vec<u8>, NetError> {
    let body = postcard::to_allocvec(frame)?;
    if FRAME_MAGIC.len() + 4 + body.len() > MAX_FRAME_BYTES {
        return Err(NetError::TooLarge(FRAME_MAGIC.len() + 4 + body.len()));
    }
    let mut out = Vec::with_capacity(FRAME_MAGIC.len() + 4 + body.len());
    out.extend_from_slice(FRAME_MAGIC);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode + validate a frame produced by [`encode_frame`].
pub fn decode_frame(bytes: &[u8]) -> Result<ActivationFrame, NetError> {
    if bytes.len() > MAX_FRAME_BYTES {
        return Err(NetError::TooLarge(bytes.len()));
    }
    if bytes.len() < FRAME_MAGIC.len() + 4 {
        return Err(NetError::BadMagic);
    }
    if &bytes[..FRAME_MAGIC.len()] != FRAME_MAGIC {
        return Err(NetError::BadMagic);
    }
    let mut len_buf = [0u8; 4];
    len_buf.copy_from_slice(&bytes[FRAME_MAGIC.len()..FRAME_MAGIC.len() + 4]);
    let declared = u32::from_le_bytes(len_buf) as usize;
    if declared > MAX_FRAME_BYTES {
        return Err(NetError::TooLarge(declared));
    }
    let actual = bytes.len() - FRAME_MAGIC.len() - 4;
    if declared != actual {
        return Err(NetError::LengthMismatch { declared, actual });
    }
    let frame: ActivationFrame =
        postcard::from_bytes(&bytes[FRAME_MAGIC.len() + 4..])?;
    Ok(frame)
}

/// Mint a self-signed dev identity via rcgen.
///
/// Returns `(cert_der, key_der)`. Persist under the exe dir / SQLite in real
/// use; tests may keep it in memory. SAN covers `dllm.local`.
pub fn dev_identity() -> Result<(Vec<u8>, Vec<u8>), NetError> {
    let certified =
        rcgen::generate_simple_self_signed(vec!["dllm.local".to_string()])?;
    let cert_der = certified.cert.der().to_vec();
    let key_der = certified.key_pair.serialize_der();
    Ok((cert_der, key_der))
}
