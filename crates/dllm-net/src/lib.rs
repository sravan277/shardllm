//! dllm-net: QUIC framing primitives (ALPN, activation frames, ACKs).
//!
//! Pure Rust only, and the wire format is **hand-rolled fixed-width
//! little-endian** — not postcard — so that non-Rust peers (the Android JNI
//! forwarder) can encode byte-for-byte from the tables in
//! `contracts/activation-frame.md` without pulling in a serialization crate.
//! The contract is authoritative; this module is its implementation, and the
//! two are kept in lockstep (see `contracts/activation-frame.md` §"Layout").

/// ALPN protocol id — must match on both ends or the handshake fails silently.
pub const ALPN: &[u8] = b"dllm/1";

/// Quinn 0.11 mTLS transport with TOFU fingerprint pinning.
pub mod transport;

/// Wire magic prefix for every activation frame, at offset 0.
pub const FRAME_MAGIC: &[u8; 5] = b"DLLM1";

/// Fixed activation-frame header size, in bytes. Total frame = `40 + payload_length`.
///
/// | Offset | Size | Field |
/// |---|---|---|
/// | 0 | 5 | `magic` = `DLLM1` |
/// | 5 | 16 | `session_id` (`u128`) |
/// | 21 | 8 | `plan_id` (`u64`) |
/// | 29 | 4 | `token_position` (`u32`) |
/// | 33 | 1 | `source_stage` (`u8`) |
/// | 34 | 1 | `target_stage` (`u8`) |
/// | 35 | 1 | `tensor_format` (`u8`) |
/// | 36 | 4 | `payload_length` (`u32`) |
pub const FRAME_HEADER_LEN: usize = 40;

/// Hard cap for a **whole** frame (`FRAME_HEADER_LEN + payload_length`).
/// Slow workers must exert backpressure instead of buffering unbounded
/// activations, so the cap is enforced *before* allocation on the decode side.
pub const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// Largest payload a single frame may carry (`8 MiB - 40`).
pub const MAX_PAYLOAD_BYTES: usize = MAX_FRAME_BYTES - FRAME_HEADER_LEN;

/// Maximum pipeline depth (`P <= 5`, per MASTER_PLAN §8 / contracts README).
pub const MAX_STAGES: usize = 5;

/// Separate magic namespace for inference ACKs. ACKs used to share
/// `FRAME_MAGIC` with activations, which made the two indistinguishable on a
/// shared stream; they now carry their own tag so a frame can never be parsed
/// as an ACK or vice versa.
pub const ACK_MAGIC: &[u8; 4] = b"ACK1";

/// ACK frame overhead: `ACK1(4) + u32-LE length(4)`.
pub const ACK_HEADER_LEN: usize = 8;

/// ACKs are tiny; anything larger is a framing error, not a big ACK.
pub const MAX_ACK_BYTES: usize = 256;

/// Tensor dtype/layout tag carried in [`ActivationFrame::tensor_format`].
///
/// `0`=fp16, `1`=fp32, `2`=q8 are real activations. `RAW` is a reserved
/// diagnostic value for opaque non-tensor payloads (plan marshalling on the
/// activation stream in the `pipe_pair` drill) and is never produced by the
/// inference path. Anything else is rejected at decode.
pub mod tensor_format {
    pub const F16: u8 = 0;
    pub const F32: u8 = 1;
    pub const Q8: u8 = 2;
    /// Reserved: opaque bytes, not a tensor. Diagnostics only.
    pub const RAW: u8 = 255;

    /// True when `v` is a format tag this build understands.
    pub fn is_known(v: u8) -> bool {
        matches!(v, F16 | F32 | Q8 | RAW)
    }
}

/// Activation tensor forwarded from one pipeline stage to the next.
///
/// Field names mirror MASTER_PLAN §8 (`session_id`, `plan_id`,
/// `token_position`, `source_stage`, `target_stage`, `tensor_format`).
/// `magic` is implicit in the codec (always [`FRAME_MAGIC`]) and
/// `payload_length` is derived from `payload`, so neither is a struct field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationFrame {
    pub session_id: u128,
    pub plan_id: u64,
    pub token_position: u32,
    pub source_stage: u8,
    pub target_stage: u8,
    pub tensor_format: u8,
    pub payload: Vec<u8>,
}

impl ActivationFrame {
    /// True when the frame satisfies the contract's edge invariants:
    /// `target_stage == source_stage + 1` and both `< MAX_STAGES`.
    pub fn has_valid_edge(&self) -> bool {
        self.source_stage < MAX_STAGES as u8
            && self.target_stage < MAX_STAGES as u8
            && self.target_stage == self.source_stage + 1
    }
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

    /// Wire vocabulary string, exactly as spelled in `contracts/acks.md`.
    pub fn name(self) -> &'static str {
        match self {
            Ack::Received { .. } => "RECEIVED",
            Ack::Computed { .. } => "COMPUTED",
            Ack::KvTentative { .. } => "KV_TENTATIVE",
            Ack::Committed { .. } => "COMMITTED",
            Ack::Truncate { .. } => "TRUNCATE",
        }
    }

    /// Flat JSON body for the control stream: `{"ack":"COMPUTED","pos":3,"token":42}`.
    /// Matches the shape named in `contracts/acks.md` §Vocabulary.
    pub fn to_wire_json(self) -> String {
        format!(
            r#"{{"ack":"{}","pos":{},"token":{}}}"#,
            self.name(),
            self.pos(),
            self.token()
        )
    }

    /// Parse the JSON body produced by [`Ack::to_wire_json`].
    pub fn from_wire_json(s: &str) -> Result<Ack, NetError> {
        #[derive(serde::Deserialize)]
        struct Wire {
            ack: String,
            pos: u32,
            token: u32,
        }
        let w: Wire = serde_json::from_str(s)?;
        let ack = match w.ack.as_str() {
            "RECEIVED" => Ack::Received { pos: w.pos, token: w.token },
            "COMPUTED" => Ack::Computed { pos: w.pos, token: w.token },
            "KV_TENTATIVE" => Ack::KvTentative { pos: w.pos, token: w.token },
            "COMMITTED" => Ack::Committed { pos: w.pos, token: w.token },
            "TRUNCATE" => Ack::Truncate { pos: w.pos, token: w.token },
            other => return Err(NetError::UnknownAck(other.to_string())),
        };
        Ok(ack)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NetError {
    #[error("frame exceeds {MAX_FRAME_BYTES} byte cap ({0} bytes)")]
    TooLarge(usize),
    #[error("bad frame magic")]
    BadMagic,
    #[error("truncated frame: need {needed} bytes, have {have}")]
    Truncated { needed: usize, have: usize },
    #[error("length prefix mismatch: declared {declared}, actual {actual}")]
    LengthMismatch { declared: usize, actual: usize },
    #[error("invalid edge: target_stage {target_stage} != source_stage {source_stage} + 1")]
    StageMismatch { source_stage: u8, target_stage: u8 },
    #[error("stage index {0} out of range (P <= {MAX_STAGES})")]
    StageOutOfRange(u8),
    #[error("unknown tensor_format {0}")]
    UnknownTensorFormat(u8),
    #[error("ACK exceeds {MAX_ACK_BYTES} byte cap ({0} bytes)")]
    AckTooLarge(usize),
    #[error("unknown ack kind {0:?}")]
    UnknownAck(String),
    #[error("malformed ack json: {0}")]
    AckJson(#[from] serde_json::Error),
    #[error("rcgen: {0}")]
    Rcgen(#[from] rcgen::Error),
}

/// Encode a frame as the contract's 40-byte fixed-width LE header + payload.
///
/// Fails with [`NetError::TooLarge`] if the total would exceed
/// [`MAX_FRAME_BYTES`], and with [`NetError::StageMismatch`] /
/// [`NetError::StageOutOfRange`] / [`NetError::UnknownTensorFormat`] if the
/// edge invariants do not hold — an invalid frame is never put on the wire.
pub fn encode_frame(frame: &ActivationFrame) -> Result<Vec<u8>, NetError> {
    if !frame.has_valid_edge() {
        if frame.source_stage >= MAX_STAGES as u8 || frame.target_stage >= MAX_STAGES as u8 {
            return Err(NetError::StageOutOfRange(
                frame.source_stage.max(frame.target_stage),
            ));
        }
        return Err(NetError::StageMismatch {
            source_stage: frame.source_stage,
            target_stage: frame.target_stage,
        });
    }
    if !tensor_format::is_known(frame.tensor_format) {
        return Err(NetError::UnknownTensorFormat(frame.tensor_format));
    }
    if frame.payload.len() > MAX_PAYLOAD_BYTES {
        return Err(NetError::TooLarge(FRAME_HEADER_LEN + frame.payload.len()));
    }

    let mut out = Vec::with_capacity(FRAME_HEADER_LEN + frame.payload.len());
    out.extend_from_slice(FRAME_MAGIC);
    out.extend_from_slice(&frame.session_id.to_le_bytes());
    out.extend_from_slice(&frame.plan_id.to_le_bytes());
    out.extend_from_slice(&frame.token_position.to_le_bytes());
    out.push(frame.source_stage);
    out.push(frame.target_stage);
    out.push(frame.tensor_format);
    out.extend_from_slice(&(frame.payload.len() as u32).to_le_bytes());
    debug_assert_eq!(out.len(), FRAME_HEADER_LEN);
    out.extend_from_slice(&frame.payload);
    Ok(out)
}

/// Serialize the fixed-width header alone (40 bytes). Lets a stream reader
/// validate the declared length *before* it allocates a payload buffer.
pub fn encode_frame_header(frame: &ActivationFrame) -> [u8; FRAME_HEADER_LEN] {
    let mut h = [0u8; FRAME_HEADER_LEN];
    h[0..5].copy_from_slice(FRAME_MAGIC);
    h[5..21].copy_from_slice(&frame.session_id.to_le_bytes());
    h[21..29].copy_from_slice(&frame.plan_id.to_le_bytes());
    h[29..33].copy_from_slice(&frame.token_position.to_le_bytes());
    h[33] = frame.source_stage;
    h[34] = frame.target_stage;
    h[35] = frame.tensor_format;
    h[36..40].copy_from_slice(&(frame.payload.len() as u32).to_le_bytes());
    h
}

/// Parse a 40-byte header, applying every magic/cap/invariant check.
///
/// Returns the header fields plus the declared payload length, which the caller
/// must have already bounded against [`MAX_PAYLOAD_BYTES`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub session_id: u128,
    pub plan_id: u64,
    pub token_position: u32,
    pub source_stage: u8,
    pub target_stage: u8,
    pub tensor_format: u8,
    pub payload_length: u32,
}

impl FrameHeader {
    /// Rebuild the header-only frame (empty payload) for validation reuse.
    pub fn as_frame(self) -> ActivationFrame {
        ActivationFrame {
            session_id: self.session_id,
            plan_id: self.plan_id,
            token_position: self.token_position,
            source_stage: self.source_stage,
            target_stage: self.target_stage,
            tensor_format: self.tensor_format,
            payload: Vec::new(),
        }
    }
}

/// Decode and fully validate a 40-byte frame header.
pub fn decode_frame_header(bytes: &[u8]) -> Result<FrameHeader, NetError> {
    if bytes.len() < FRAME_HEADER_LEN {
        return Err(NetError::Truncated {
            needed: FRAME_HEADER_LEN,
            have: bytes.len(),
        });
    }
    if &bytes[0..5] != FRAME_MAGIC {
        return Err(NetError::BadMagic);
    }
    let mut sid = [0u8; 16];
    sid.copy_from_slice(&bytes[5..21]);
    let mut pid = [0u8; 8];
    pid.copy_from_slice(&bytes[21..29]);
    let mut tpos = [0u8; 4];
    tpos.copy_from_slice(&bytes[29..33]);
    let hdr = FrameHeader {
        session_id: u128::from_le_bytes(sid),
        plan_id: u64::from_le_bytes(pid),
        token_position: u32::from_le_bytes(tpos),
        source_stage: bytes[33],
        target_stage: bytes[34],
        tensor_format: bytes[35],
        payload_length: u32::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]),
    };
    // Size is checked before any semantic interpretation: a hostile header must
    // be rejected on its declared length without us trusting the rest of it.
    if hdr.payload_length as usize > MAX_PAYLOAD_BYTES {
        return Err(NetError::TooLarge(
            FRAME_HEADER_LEN + hdr.payload_length as usize,
        ));
    }
    // Invariants are checked at header-parse time so a stream reader rejects a
    // bad edge before it ever reserves a payload buffer.
    if hdr.source_stage >= MAX_STAGES as u8 || hdr.target_stage >= MAX_STAGES as u8 {
        return Err(NetError::StageOutOfRange(
            hdr.source_stage.max(hdr.target_stage),
        ));
    }
    if hdr.target_stage != hdr.source_stage + 1 {
        return Err(NetError::StageMismatch {
            source_stage: hdr.source_stage,
            target_stage: hdr.target_stage,
        });
    }
    if !tensor_format::is_known(hdr.tensor_format) {
        return Err(NetError::UnknownTensorFormat(hdr.tensor_format));
    }
    Ok(hdr)
}

/// Decode + validate a complete frame (header + payload) produced by
/// [`encode_frame`].
pub fn decode_frame(bytes: &[u8]) -> Result<ActivationFrame, NetError> {
    if bytes.len() > MAX_FRAME_BYTES {
        return Err(NetError::TooLarge(bytes.len()));
    }
    let hdr = decode_frame_header(bytes)?;
    let declared = hdr.payload_length as usize;
    let actual = bytes.len() - FRAME_HEADER_LEN;
    if declared != actual {
        return Err(NetError::LengthMismatch { declared, actual });
    }
    let mut frame = hdr.as_frame();
    frame.payload = bytes[FRAME_HEADER_LEN..].to_vec();
    Ok(frame)
}

/// Encode an ACK as `ACK1(4) || u32-LE json_len || {"ack":…}`.
///
/// Uses its own magic so an ACK can never collide with an activation frame on a
/// shared stream (they previously shared `DLLM1`).
pub fn encode_ack(ack: &Ack) -> Result<Vec<u8>, NetError> {
    let body = ack.to_wire_json().into_bytes();
    if body.len() > MAX_ACK_BYTES {
        return Err(NetError::AckTooLarge(body.len()));
    }
    let mut out = Vec::with_capacity(ACK_HEADER_LEN + body.len());
    out.extend_from_slice(ACK_MAGIC);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode an ACK framed by [`encode_ack`]. Length is cap-checked before the
/// body is parsed.
pub fn decode_ack(bytes: &[u8]) -> Result<Ack, NetError> {
    if bytes.len() < ACK_HEADER_LEN {
        return Err(NetError::Truncated {
            needed: ACK_HEADER_LEN,
            have: bytes.len(),
        });
    }
    if &bytes[0..4] != ACK_MAGIC {
        return Err(NetError::BadMagic);
    }
    let declared =
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if declared > MAX_ACK_BYTES {
        return Err(NetError::AckTooLarge(declared));
    }
    let body = &bytes[ACK_HEADER_LEN..];
    if body.len() != declared {
        return Err(NetError::LengthMismatch {
            declared,
            actual: body.len(),
        });
    }
    let text = std::str::from_utf8(body).map_err(|_| NetError::UnknownAck("<non-utf8>".into()))?;
    Ack::from_wire_json(text)
}

/// Mint a self-signed dev identity via rcgen.
///
/// Returns `(cert_der, key_der)`. Persist under the exe dir / SQLite in real
/// use; tests may keep it in memory. SAN covers `dllm.local`.
pub fn dev_identity() -> Result<(Vec<u8>, Vec<u8>), NetError> {
    let certified =
        rcgen::generate_simple_self_signed(vec!["dllm.local".to_string()])?;
    let cert_der = certified.cert.der().to_vec();
    let key_der = certified.signing_key.serialize_der();
    Ok((cert_der, key_der))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(payload: Vec<u8>) -> ActivationFrame {
        ActivationFrame {
            session_id: 0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10,
            plan_id: 0x1122_3344_5566_7788,
            token_position: 0xdead_beef,
            source_stage: 0,
            target_stage: 1,
            tensor_format: tensor_format::F16,
            payload,
        }
    }

    #[test]
    fn header_is_forty_bytes_with_contract_field_offsets() {
        let wire = encode_frame(&frame(vec![0u8; 2048])).unwrap();
        assert_eq!(wire.len(), FRAME_HEADER_LEN + 2048);
        assert_eq!(&wire[0..5], b"DLLM1");
        assert_eq!(u128::from_le_bytes(wire[5..21].try_into().unwrap()), 0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10);
        assert_eq!(u64::from_le_bytes(wire[21..29].try_into().unwrap()), 0x1122_3344_5566_7788);
        assert_eq!(u32::from_le_bytes(wire[29..33].try_into().unwrap()), 0xdead_beef);
        assert_eq!(wire[33], 0, "source_stage @33");
        assert_eq!(wire[34], 1, "target_stage @34");
        assert_eq!(wire[35], tensor_format::F16, "tensor_format @35");
        assert_eq!(u32::from_le_bytes(wire[36..40].try_into().unwrap()), 2048, "payload_length @36");
    }

    #[test]
    fn header_helper_matches_full_encoder() {
        let f = frame(vec![7u8; 10]);
        let h = encode_frame_header(&f);
        assert_eq!(&h[..], &encode_frame(&f).unwrap()[..FRAME_HEADER_LEN]);
        assert_eq!(decode_frame_header(&h).unwrap().as_frame().payload, Vec::<u8>::new());
    }

    #[test]
    fn rejects_non_adjacent_stage_edge() {
        let mut f = frame(vec![1, 2, 3]);
        f.target_stage = 3;
        assert!(matches!(encode_frame(&f), Err(NetError::StageMismatch { source_stage: 0, target_stage: 3 })));
    }

    #[test]
    fn rejects_stage_index_at_or_above_p_max() {
        let mut f = frame(vec![1]);
        f.source_stage = 4;
        f.target_stage = 5; // would be adjacent, but 5 >= MAX_STAGES
        assert!(matches!(encode_frame(&f), Err(NetError::StageOutOfRange(5))));
    }

    #[test]
    fn rejects_unknown_tensor_format() {
        let mut f = frame(vec![1]);
        f.tensor_format = 7;
        assert!(matches!(encode_frame(&f), Err(NetError::UnknownTensorFormat(7))));
    }

    #[test]
    fn decode_rejects_truncated_header() {
        let wire = encode_frame(&frame(vec![9; 4])).unwrap();
        assert!(matches!(
            decode_frame(&wire[..FRAME_HEADER_LEN - 1]),
            Err(NetError::Truncated { .. })
        ));
    }

    #[test]
    fn ack_roundtrips_and_uses_its_own_magic() {
        for ack in [
            Ack::Received { pos: 1, token: 2 },
            Ack::Computed { pos: 3, token: 4 },
            Ack::KvTentative { pos: 5, token: 6 },
            Ack::Committed { pos: 7, token: 8 },
            Ack::Truncate { pos: 9, token: 10 },
        ] {
            let wire = encode_ack(&ack).unwrap();
            assert_eq!(&wire[0..4], ACK_MAGIC);
            assert_ne!(&wire[0..5], &FRAME_MAGIC[..], "ack must not look like a frame");
            assert_eq!(decode_ack(&wire).unwrap(), ack);
        }
    }

    #[test]
    fn ack_json_uses_contract_vocabulary_strings() {
        let json = Ack::KvTentative { pos: 42, token: 7 }.to_wire_json();
        assert_eq!(json, r#"{"ack":"KV_TENTATIVE","pos":42,"token":7}"#);
        assert_eq!(Ack::from_wire_json(&json).unwrap(), Ack::KvTentative { pos: 42, token: 7 });
        assert!(matches!(
            Ack::from_wire_json(r#"{"ack":"NOPE","pos":1,"token":1}"#),
            Err(NetError::UnknownAck(_))
        ));
    }

    #[test]
    fn oversized_payload_rejected_on_encode() {
        let f = frame(vec![0u8; MAX_FRAME_BYTES]);
        assert!(matches!(encode_frame(&f), Err(NetError::TooLarge(_))));
    }
}