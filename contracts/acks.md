# Inference ACK vocabulary + optimistic commit flow

Transport ACK (QUIC/TCP) ≠ inference ACK. This file defines inference ACKs
only. All inference ACKs ride the reliable control stream; activations ride
the per-edge stream (see `activation-frame.md`).

## Vocabulary (exact strings on the wire, JSON `{"ack": …}`)

| ACK | From → to | Meaning |
|---|---|---|
| `RECEIVED(pos)` | stage → coordinator | Frame for `pos` passed framing checks, queued |
| `COMPUTED(pos)` | stage → coordinator | Layers done, `KV_TENTATIVE(pos)` appended locally, activation forwarded downstream |
| `KV_TENTATIVE(pos)` | stage → coordinator | Tentative KV rows for `pos` durable in stage-local pages (sent together with `COMPUTED`; separate variant so storage vs compute failures are distinguishable) |
| `COMMITTED(pos)` | coordinator → all stages + clients | `pos` is final. Emitted only after **every** stage reported `COMPUTED+KV_TENTATIVE(pos)` |
| `TRUNCATE(pos)` | coordinator → all stages | Discard tentative `pos, pos+1, …`. The stream rewinds to last committed |

`pos` = `token_position` (`u32`, 0-based, dense per session).

## Optimistic primary-backup flow (no per-token 2PC)

```text
steady state (no extra RTT):
  coord --dispatch(pos, COMMIT(pos-1) piggyback)--> S0 --> S1 --> S2(tail)
  S0,S1,S2 --COMPUTED+KV_TENTATIVE(pos)--> coord   (async, out of band)
  S2 --token_id(pos)--> coord
  coord --COMMITTED(pos) piggybacked on dispatch(pos+1)--> stages
  coord --SSE token(pos)--> clients (only after COMMITTED)

abort:
  coord --TRUNCATE(pos)--> all stages (drop tentative >= pos, keep <= pos-1)
  clients never saw pos (only COMMITTED tokens are streamed)
```

- Stages append tentatively **without waiting**. Coordinator piggybacks
  `COMMIT(pos-1)` on the next dispatch; a standalone `COMMITTED` is sent only
  when the pipeline idles.
- Recovery replays from last `COMMITTED` (or `checkpoint_K` snapshot);
  tentative rows are never checkpointed, never streamed, never backed up.

## seq/pos assignment

- Coordinator owns numbering. One `u32` counter per `session_id`, starts at
  `len(committed transcript)` after prefill (prompt tokens occupy `0..n`).
- Prefill chunks reserve ranges atomically (`pos..pos+C`); decode reserves one.
- Retries reuse the same `pos` (idempotent: same `(session, plan, pos,
  token_id)` → same KV bytes). A new `plan_id` restarts tentative state but
  keeps committed `pos` numbering.
- Stale `plan_id` frames/ACKs are dropped and counted.

## Single tail sampler rule

- Sampling happens **exactly once, at the tail stage**, with the session's
  pinned sampler params (thinking mode + seed from session creation).
- No per-stage sampling, no logit shipping (full vocab never crosses the LAN).
- Log top-1 margin per token for the cross-ISA watchdog
  (`max|Δlogit|<0.5`, top-1 >99.5% vs single-device baseline).
