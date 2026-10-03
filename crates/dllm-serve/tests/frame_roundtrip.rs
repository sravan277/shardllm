//! Activation-frame + ACK wire-format property tests.
//!
//! The activation frame is the cross-target wire format, so these tests assert
//! the contract byte-for-byte: fixed-width little-endian 40-byte header with
//! `magic` at offset 0, payload `length` at offset 36, `40 + payload_length <= 8 MiB`.
//! A non-Rust peer (Android JNI) hand-encodes from `contracts/activation-frame.md`,
//! so field offsets are pinned here rather than left to a serialization crate.

use dllm_net::tensor_format;
use dllm_net::{
    ACK_HEADER_LEN, ACK_MAGIC, FRAME_HEADER_LEN, FRAME_MAGIC, MAX_ACK_BYTES, MAX_FRAME_BYTES,
    MAX_PAYLOAD_BYTES, MAX_STAGES, Ack, ActivationFrame, NetError, decode_ack, decode_frame,
    decode_frame_header, encode_ack, encode_frame, encode_frame_header,
};

/// Deterministic xorshift64 — no `rand` dependency for one property test.
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        while v.len() < len {
            v.extend_from_slice(&self.next().to_le_bytes());
        }
        v.truncate(len);
        v
    }
}

/// Random payload plus random scalars, but always a **contract-valid edge**
/// (`target_stage == source_stage + 1`, both `< P`).
fn random_frame(payload: Vec<u8>, rng: &mut XorShift64, format: u8) -> ActivationFrame {
    let source_stage = (rng.next() as usize % (MAX_STAGES - 1)) as u8;
    ActivationFrame {
        session_id: (u128::from(rng.next()) << 64) | u128::from(rng.next()),
        plan_id: rng.next(),
        token_position: (rng.next() >> 32) as u32,
        source_stage,
        target_stage: source_stage + 1,
        tensor_format: format,
        payload,
    }
}

/// Small deterministic scalars so a near-cap payload still fits deterministically.
fn minimal_frame(payload: Vec<u8>, format: u8) -> ActivationFrame {
    ActivationFrame {
        session_id: 1,
        plan_id: 1,
        token_position: 0,
        source_stage: 0,
        target_stage: 1,
        tensor_format: format,
        payload,
    }
}

#[test]
fn activation_frame_roundtrips_across_payload_sizes() {
    const OVERHEAD: usize = FRAME_HEADER_LEN;
    assert_eq!(OVERHEAD, 40, "contract fixes the header at 40 bytes");
    assert_eq!(MAX_PAYLOAD_BYTES, MAX_FRAME_BYTES - FRAME_HEADER_LEN);

    // 0 B .. just under the 8 MiB cap, spread across sizes.
    const SIZES: [usize; 10] = [
        0,
        1,
        63,
        1 << 10,
        1 << 16,
        1 << 20,
        4_000_000,
        7_000_000,
        MAX_PAYLOAD_BYTES - 8,
        123_456,
    ];
    let formats = [
        tensor_format::F16,
        tensor_format::F32,
        tensor_format::Q8,
        tensor_format::RAW,
    ];

    let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
    for (i, &size) in SIZES.iter().enumerate() {
        let f = random_frame(rng.bytes(size), &mut rng, formats[i % formats.len()]);
        let wire = encode_frame(&f).unwrap_or_else(|e| panic!("size {size}: encode: {e}"));
        assert_eq!(
            wire.len(),
            FRAME_HEADER_LEN + size,
            "total frame must be exactly 40 + payload_length"
        );
        assert!(wire.len() <= MAX_FRAME_BYTES);
        let back = decode_frame(&wire).unwrap_or_else(|e| panic!("size {size}: decode: {e}"));
        assert_eq!(back, f, "round-trip mismatch at payload size {size}");
    }
}

#[test]
fn header_layout_matches_contract_field_offsets() {
    let f = ActivationFrame {
        session_id: 0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10,
        plan_id: 0x1122_3344_5566_7788,
        token_position: 0x0bad_f00d,
        source_stage: 2,
        target_stage: 3,
        tensor_format: tensor_format::Q8,
        payload: vec![0u8; 2048],
    };
    let wire = encode_frame(&f).unwrap();

    assert_eq!(&wire[0..5], b"DLLM1", "offset 0: magic");
    assert_eq!(u128::from_le_bytes(wire[5..21].try_into().unwrap()), f.session_id, "offset 5: session_id");
    assert_eq!(u64::from_le_bytes(wire[21..29].try_into().unwrap()), f.plan_id, "offset 21: plan_id");
    assert_eq!(u32::from_le_bytes(wire[29..33].try_into().unwrap()), f.token_position, "offset 29: token_position");
    assert_eq!(wire[33], 2, "offset 33: source_stage");
    assert_eq!(wire[34], 3, "offset 34: target_stage");
    assert_eq!(wire[35], tensor_format::Q8, "offset 35: tensor_format");
    assert_eq!(u32::from_le_bytes(wire[36..40].try_into().unwrap()), 2048, "offset 36: payload_length");

    // Header-only encode must be byte-identical to the full encoder's prefix,
    // so a streaming reader and a buffering decoder agree.
    assert_eq!(&wire[..FRAME_HEADER_LEN], &encode_frame_header(&f)[..]);

    // Header parse alone (no payload) still enforces every invariant.
    let hdr = decode_frame_header(&wire[..FRAME_HEADER_LEN]).unwrap();
    assert_eq!(hdr.as_frame().source_stage, 2);
    assert_eq!(hdr.payload_length, 2048);
}

#[test]
fn oversized_frames_are_rejected_on_both_sides() {
    let mut rng = XorShift64(0x0DDB_1A75_1EED_5EED);

    // Largest legal payload still encodes and round-trips.
    let at_cap = minimal_frame(rng.bytes(MAX_PAYLOAD_BYTES), tensor_format::Q8);
    let wire = encode_frame(&at_cap).expect("frame exactly at the cap encodes");
    assert_eq!(wire.len(), MAX_FRAME_BYTES);
    assert_eq!(decode_frame(&wire).unwrap(), at_cap);

    // One payload byte over the budget is rejected on encode.
    let huge = minimal_frame(vec![0u8; MAX_FRAME_BYTES], tensor_format::F32);
    assert!(matches!(encode_frame(&huge), Err(NetError::TooLarge(_))));

    // Decode side: input buffer beyond the cap is rejected before parsing.
    let oversized = vec![0u8; MAX_FRAME_BYTES + 1];
    assert!(matches!(decode_frame(&oversized), Err(NetError::TooLarge(_))));

    // A huge declared payload_length in the header is rejected without reading a body.
    let mut evil = vec![0u8; FRAME_HEADER_LEN];
    evil[..5].copy_from_slice(FRAME_MAGIC);
    evil[36..40].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(decode_frame_header(&evil), Err(NetError::TooLarge(_))));

    // Malformed frames: corrupt magic, truncated header, length mismatch.
    let mut bad_magic = encode_frame(&minimal_frame(vec![1, 2, 3], tensor_format::F32)).unwrap();
    bad_magic[0] ^= 0xFF;
    assert!(matches!(decode_frame(&bad_magic), Err(NetError::BadMagic)));

    let mut short = encode_frame(&minimal_frame(vec![1, 2, 3], tensor_format::F32)).unwrap();
    short.truncate(FRAME_HEADER_LEN - 1);
    assert!(matches!(
        decode_frame(&short),
        Err(NetError::Truncated { .. })
    ));

    let mut padded = encode_frame(&minimal_frame(vec![1, 2, 3], tensor_format::F32)).unwrap();
    padded.push(0);
    assert!(matches!(
        decode_frame(&padded),
        Err(NetError::LengthMismatch { .. })
    ));
}

#[test]
fn edge_invariants_are_enforced_not_just_documented() {
    // target_stage must be exactly source_stage + 1.
    let mut skip = minimal_frame(vec![1], tensor_format::F16);
    skip.target_stage = 3;
    assert!(matches!(
        encode_frame(&skip),
        Err(NetError::StageMismatch { source_stage: 0, target_stage: 3 })
    ));
    let mut backwards = minimal_frame(vec![1], tensor_format::F16);
    backwards.source_stage = 2;
    backwards.target_stage = 1;
    assert!(matches!(
        encode_frame(&backwards),
        Err(NetError::StageMismatch { source_stage: 2, target_stage: 1 })
    ));

    // Both indices must be < P (P <= 5).
    let mut past_p = minimal_frame(vec![1], tensor_format::F16);
    past_p.source_stage = (MAX_STAGES - 1) as u8;
    past_p.target_stage = MAX_STAGES as u8;
    assert!(matches!(encode_frame(&past_p), Err(NetError::StageOutOfRange(_))));
    let mut way_past = minimal_frame(vec![1], tensor_format::F16);
    way_past.source_stage = 200;
    way_past.target_stage = 201;
    assert!(matches!(encode_frame(&way_past), Err(NetError::StageOutOfRange(201))));

    // Unknown tensor_format is dropped, not silently accepted.
    let mut bad_fmt = minimal_frame(vec![1], tensor_format::F16);
    bad_fmt.tensor_format = 9;
    assert!(matches!(encode_frame(&bad_fmt), Err(NetError::UnknownTensorFormat(9))));

    // The last legal edge (stage 3 -> 4) is accepted, proving the bound is off-by-one correct.
    let last = ActivationFrame { source_stage: 3, target_stage: 4, ..minimal_frame(vec![1], tensor_format::F16) };
    assert!(encode_frame(&last).is_ok());

    // A hostile header carrying a bad edge is rejected at header-parse time,
    // i.e. before a payload buffer is ever reserved.
    let mut hostile = encode_frame(&minimal_frame(vec![0u8; 1024], tensor_format::F16)).unwrap();
    hostile[34] = 2; // target_stage in range but not source+1
    assert!(matches!(decode_frame_header(&hostile), Err(NetError::StageMismatch { .. })));
    let mut hostile_range = hostile.clone();
    hostile_range[34] = 9; // target_stage >= P
    assert!(matches!(decode_frame_header(&hostile_range), Err(NetError::StageOutOfRange(_))));

    // Size is validated before semantics: an oversized declared length is
    // rejected on the cap alone, not on whatever the other fields happen to say.
    let mut evil_sized = encode_frame(&minimal_frame(vec![0u8; 16], tensor_format::F16)).unwrap();
    evil_sized[36..40].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(decode_frame_header(&evil_sized), Err(NetError::TooLarge(_))));
}

#[test]
fn acks_roundtrip_on_their_own_magic_namespace() {
    let all = [
        Ack::Received { pos: 0, token: 11 },
        Ack::Computed { pos: 1, token: 22 },
        Ack::KvTentative { pos: 2, token: 33 },
        Ack::Committed { pos: 3, token: 44 },
        Ack::Truncate { pos: 4, token: 55 },
    ];
    for ack in all {
        let wire = encode_ack(&ack).unwrap();
        assert_eq!(&wire[0..4], ACK_MAGIC, "ACK1 magic");
        assert_ne!(
            &wire[0..5],
            &FRAME_MAGIC[..],
            "an ACK must never be mistaken for an activation frame"
        );
        assert_eq!(wire.len(), ACK_HEADER_LEN + wire[ACK_HEADER_LEN..].len());
        assert!(wire.len() <= ACK_HEADER_LEN + MAX_ACK_BYTES);
        assert_eq!(decode_ack(&wire).unwrap(), ack);
    }

    // Vocabulary strings match contracts/acks.md exactly.
    assert_eq!(Ack::Received { pos: 0, token: 0 }.name(), "RECEIVED");
    assert_eq!(Ack::Computed { pos: 0, token: 0 }.name(), "COMPUTED");
    assert_eq!(Ack::KvTentative { pos: 0, token: 0 }.name(), "KV_TENTATIVE");
    assert_eq!(Ack::Committed { pos: 0, token: 0 }.name(), "COMMITTED");
    assert_eq!(Ack::Truncate { pos: 0, token: 0 }.name(), "TRUNCATE");
    assert_eq!(
        Ack::KvTentative { pos: 7, token: 9 }.to_wire_json(),
        r#"{"ack":"KV_TENTATIVE","pos":7,"token":9}"#
    );

    // Truncated / corrupt ACK frames error instead of panicking.
    let good = encode_ack(&Ack::Committed { pos: 1, token: 2 }).unwrap();
    assert!(matches!(decode_ack(&good[..3]), Err(NetError::Truncated { .. })));
    let mut bad = good.clone();
    bad[0] ^= 0xFF;
    assert!(matches!(decode_ack(&bad), Err(NetError::BadMagic)));
    assert!(matches!(
        decode_ack(&good[..good.len() - 2]),
        Err(NetError::LengthMismatch { .. })
    ));
}