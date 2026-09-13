//! Activation-frame wire-format property tests (10 random frames, payloads
//! 0 B .. 8 MiB cap, oversized rejection on both sides).
//!
//! Lives in `dllm-serve` because `dllm-net` is off-limits to edits in this
//! change; `dllm-net` is used directly as a dev-dependency.

use dllm_net::tensor_format;
use dllm_net::{decode_frame, encode_frame, ActivationFrame, NetError, FRAME_MAGIC, MAX_FRAME_BYTES};

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

fn random_frame(payload: Vec<u8>, rng: &mut XorShift64, tensor_format: u8) -> ActivationFrame {
    ActivationFrame {
        session_id: (u128::from(rng.next()) << 64) | u128::from(rng.next()),
        plan_id: rng.next(),
        token_position: (rng.next() >> 32) as u32,
        source_stage: rng.next() as u8,
        target_stage: (rng.next() >> 8) as u8,
        tensor_format,
        payload,
    }
}

/// Small deterministic scalars so the frame overhead is fixed and a payload
/// near the 8 MiB wire cap fits deterministically.
fn minimal_frame(payload: Vec<u8>, tensor_format: u8) -> ActivationFrame {
    ActivationFrame {
        session_id: 1,
        plan_id: 1,
        token_position: 0,
        source_stage: 0,
        target_stage: 1,
        tensor_format,
        payload,
    }
}

#[test]
fn activation_frame_roundtrips_across_payload_sizes() {
    // MAGIC(5) + u32-LE length prefix (4) = 9 bytes of framing overhead.
    const OVERHEAD: usize = FRAME_MAGIC.len() + 4;
    assert_eq!(OVERHEAD, 9);

    // 0 B .. 8 MiB spread; the largest (8_000_000 B) stays safely below the
    // wire cap even with maximal postcard varint widths.
    const SIZES: [usize; 10] = [
        0,
        1,
        63,
        1 << 10,
        1 << 16,
        1 << 20,
        4_000_000,
        7_000_000,
        8_000_000,
        123_456,
    ];
    let formats = [
        tensor_format::F16,
        tensor_format::F32,
        tensor_format::Q8,
        tensor_format::U8_BYTES,
    ];

    let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
    for (i, &size) in SIZES.iter().enumerate() {
        let f = random_frame(rng.bytes(size), &mut rng, formats[i % formats.len()]);
        let wire = encode_frame(&f).unwrap_or_else(|e| panic!("size {size}: encode: {e}"));
        assert!(wire.len() > size && wire.len() <= MAX_FRAME_BYTES);
        let back =
            decode_frame(&wire).unwrap_or_else(|e| panic!("size {size}: decode: {e}"));
        assert_eq!(back, f, "round-trip mismatch at payload size {size}");
    }
}

#[test]
fn oversized_frames_are_rejected_on_both_sides() {
    let mut rng = XorShift64(0x0DDB_1A75_1EED_5EED);

    // Just under the cap (8 MiB - 100 bytes of payload with fixed small
    // scalars) still encodes and round-trips.
    let near_cap = minimal_frame(rng.bytes(MAX_FRAME_BYTES - 100), tensor_format::Q8);
    let wire = encode_frame(&near_cap).expect("frame just under the cap encodes");
    assert!(wire.len() <= MAX_FRAME_BYTES);
    assert_eq!(decode_frame(&wire).unwrap(), near_cap);

    // One payload over the 8 MiB budget is rejected on encode, no matter
    // how small the scalar fields are.
    let huge = minimal_frame(vec![0u8; MAX_FRAME_BYTES], tensor_format::F32);
    assert!(matches!(encode_frame(&huge), Err(NetError::TooLarge(_))));

    // Decode side: input buffer beyond the cap is rejected before parsing.
    let oversized = vec![0u8; MAX_FRAME_BYTES + 1];
    assert!(matches!(decode_frame(&oversized), Err(NetError::TooLarge(_))));

    // A huge declared length is rejected without reading the body.
    let mut evil = Vec::new();
    evil.extend_from_slice(FRAME_MAGIC);
    evil.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(decode_frame(&evil), Err(NetError::TooLarge(_))));

    // Malformed frames: corrupt magic, truncated buffer, length mismatch.
    let mut bad_magic = encode_frame(&minimal_frame(vec![1, 2, 3], tensor_format::F32)).unwrap();
    bad_magic[0] ^= 0xFF;
    assert!(matches!(decode_frame(&bad_magic), Err(NetError::BadMagic)));

    let mut short = encode_frame(&minimal_frame(vec![1, 2, 3], tensor_format::F32)).unwrap();
    short.truncate(FRAME_MAGIC.len() + 3);
    assert!(matches!(decode_frame(&short), Err(NetError::BadMagic)));

    let mut padded = encode_frame(&minimal_frame(vec![1, 2, 3], tensor_format::F32)).unwrap();
    padded.push(0);
    assert!(matches!(
        decode_frame(&padded),
        Err(NetError::LengthMismatch { .. })
    ));
}