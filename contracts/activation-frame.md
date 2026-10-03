# ActivationFrame v1 — binary spec

Stage-to-stage activation carrier. One frame = one QUIC/TCP stream message
for one `(session, plan, token_position)` edge. Little-endian throughout.

## Layout (fixed-width little-endian, encode/decode in this exact order)

| Offset | Size | Field | Type | Notes |
|---|---|---|---|---|
| 0 | 5 | `magic` | `u8[5]` | ASCII `DLLM1` (`44 4C 4C 4D 31`). Reject anything else |
| 5 | 16 | `session_id` | `u128` | Coordinator-issued, UUID bytes LE |
| 21 | 8 | `plan_id` | `u64` | Immutable `pipeline_plan_id` for this session |
| 29 | 4 | `token_position` | `u32` | 0-based position in committed+tentative sequence |
| 33 | 1 | `source_stage` | `u8` | Sender index (0-based, `P<=5`) |
| 34 | 1 | `target_stage` | `u8` | Must equal `source_stage + 1` |
| 35 | 1 | `tensor_format` | `u8` enum | `0`=fp16, `1`=fp32, `2`=q8, `255`=RAW (reserved, see below) |
| 36 | 4 | `payload_length` | `u32` | Bytes of `payload` that follow |
| 40 | N | `payload` | `u8[N]` | Row-major hidden states `[batch, seq, 1024]` in `tensor_format` |

Header = 40 bytes. Total frame = `40 + payload_length`.

```rust
// crates/dllm-net — canonical shape. `magic` is implicit in the codec (always
// DLLM1) and `payload_length` is derived from `payload`, so neither is a field.
pub struct ActivationFrame {
    pub session_id: u128,
    pub plan_id: u64,
    pub token_position: u32,
    pub source_stage: u8,
    pub target_stage: u8,
    pub tensor_format: u8,
    pub payload: Vec<u8>,
}
```

## Encoding rules (NOT a serialization framework)

Every integer above is **fixed-width little-endian**. There is no varint, no
length-prefixed field, and **no postcard**. This is deliberate: the Android JNI
forwarder and any future non-Rust peer must be able to encode byte-for-byte from
this table with nothing but `ByteBuffer.putLong/putInt/put`, so the format must
not depend on a Rust-only encoding rule.

Consequences that used to be violated and are now enforced by
`crates/dllm-serve/tests/frame_roundtrip.rs`:

- `magic` is **field 0 at offset 0**, not a prefix outside the body.
- `payload_length` is an **explicit `u32` at offset 36**, not a `Vec<u8>` length prefix.
- Total frame is **always** `40 + payload_length`, so the framing overhead is a
  constant 40 bytes regardless of payload size or field magnitudes.

## Limits

- **Frame cap 8 MiB**: `40 + payload_length <= 8_388_608`. Sender must chunk
  prefill so no frame exceeds it; receiver drops + counts oversize frames.
- `target_stage == source_stage + 1`, both `< 5`. Unknown `tensor_format` → drop.
- `payload_length` must equal actual trailing bytes; short/long = framing error,
  tear down the edge stream, keep control stream alive.
- Prefill chunking (SLO knob `prefill_chunk`, 256–512) is sized so
  `chunk_tokens × 1024 × elem_size ≤ cap` per edge.

### Validation order (must be cheap-before-semantic)

A receiver parses in this order so a hostile or corrupt peer can never cause a
large allocation or reach semantic checks on untrusted data:

1. enough bytes for the 40-byte header → else `Truncated`
2. `magic == DLLM1` → else `BadMagic`
3. `payload_length` against the 8 MiB cap → else `TooLarge`
4. `source_stage`/`target_stage` `< 5` → else `StageOutOfRange`
5. `target_stage == source_stage + 1` → else `StageMismatch`
6. `tensor_format` known → else `UnknownTensorFormat`
7. trailing byte count `== payload_length` → else `LengthMismatch`

Steps 1–6 operate on the header alone, so a stream reader can reject a bad frame
**before** reserving the payload buffer. `decode_frame_header` exposes exactly
this, and `encode_frame_header`/`encode_frame` share one byte layout.

## `tensor_format` values

| Value | Meaning |
|---|---|
| `0` | fp16 — the MVP decode path |
| `1` | fp32 |
| `2` | q8 — same `[batch, seq, 1024]` logical shape |
| `255` | `RAW` — **reserved, diagnostics only**: opaque non-tensor bytes. Used to marshal a plan/control JSON on the activation stream in the `pipe_pair` drill. The inference path never emits `RAW`. |

Anything else is `UnknownTensorFormat` and the frame is dropped.

## Payload convention

- Decode (1 token): `[1, 1, 1024]` fp16 = 2048 B.
- Prefill chunk of C tokens: `[1, C, 1024]` fp16 = `2048 × C` B.
- Batching concatenates on `seq` dim; `token_position` = position of the
  **first** token in the frame; per-token positions are dense from there.
- Quantized `q8` payloads keep identical `[batch, seq, 1024]` logical shape;
  scale/zero metadata, if any, is defined by a future v1.1 (negotiate via control).

## ACK framing is a separate namespace

Inference ACKs ride the control stream and are **not** activation frames. They
carry their own magic `ACK1` (`41 43 4B 31`) so the two can never be confused
on a shared stream — see `acks.md` for the ACK frame layout. `ACK1` and `DLLM1`
are distinct namespaces; an `ACK1` buffer handed to the frame decoder is
rejected as `BadMagic`.

Bump `magic` to `DLLM2` for any layout change; never version inside the frame.
