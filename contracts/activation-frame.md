# ActivationFrame v1 — binary spec

Stage-to-stage activation carrier. One frame = one QUIC/TCP stream message
for one `(session, plan, token_position)` edge. Little-endian throughout.

## Layout (encode/decode in this exact order — postcard-compatible)

| Offset | Size | Field | Type | Notes |
|---|---|---|---|---|
| 0 | 5 | `magic` | `u8[5]` | ASCII `DLLM1` (`44 4C 4C 4D 31`). Reject anything else |
| 5 | 16 | `session_id` | `u128` | Coordinator-issued, UUID bytes LE |
| 21 | 8 | `plan_id` | `u64` | Immutable `pipeline_plan_id` for this session |
| 29 | 4 | `token_position` | `u32` | 0-based position in committed+tentative sequence |
| 33 | 1 | `source_stage` | `u8` | Sender index (0-based, `P<=5`) |
| 34 | 1 | `target_stage` | `u8` | Must equal `source_stage + 1` |
| 35 | 1 | `tensor_format` | `u8` enum | `0`=fp16, `1`=fp32, `2`=q8. MVP decode path uses `0` |
| 36 | 4 | `payload_length` | `u32` | Bytes of `payload` that follow |
| 40 | N | `payload` | `u8[N]` | Row-major hidden states `[batch, seq, 1024]` in `tensor_format` |

Header = 40 bytes. Total frame = `40 + payload_length`.

```rust
// crates/protocol — canonical shape (postcard field order = table order)
pub struct ActivationFrame {
    pub magic: [u8; 5],      // b"DLLM1"
    pub session_id: u128,    // le
    pub plan_id: u64,        // le
    pub token_position: u32, // le
    pub source_stage: u8,
    pub target_stage: u8,
    pub tensor_format: TensorFormat, // u8: 0 fp16 | 1 fp32 | 2 q8
    pub payload_length: u32, // le, == payload.len()
    pub payload: Vec<u8>,
}
```

## Limits

- **Frame cap 8 MiB**: `40 + payload_length <= 8_388_608`. Sender must chunk
  prefill so no frame exceeds it; receiver drops + counts oversize frames.
- `target_stage == source_stage + 1`, both `< 5`. Unknown `tensor_format` → drop.
- `payload_length` must equal actual trailing bytes; short/long = framing error,
  tear down the edge stream, keep control stream alive.
- Prefill chunking (SLO knob `prefill_chunk`, 256–512) is sized so
  `chunk_tokens × 1024 × elem_size ≤ cap` per edge.

## Payload convention

- Decode (1 token): `[1, 1, 1024]` fp16 = 2048 B.
- Prefill chunk of C tokens: `[1, C, 1024]` fp16 = `2048 × C` B.
- Batching concatenates on `seq` dim; `token_position` = position of the
  **first** token in the frame; per-token positions are dense from there.
- Quantized `q8` payloads keep identical `[batch, seq, 1024]` logical shape;
  scale/zero metadata, if any, is defined by a future v1.1 (negotiate via control).

## Postcard note

Field order above IS the postcard serialization order (`magic` first,
`payload` last as `Vec<u8>` with its `u32` length prefix = `payload_length`).
Rust impl derives `Serialize/Deserialize` on the struct in this order; non-Rust
peers (Android JNI) hand-encode LE ints to match — no postcard lib required.
Bump `magic` to `DLLM2` for any layout change; never version inside the frame.
