# Research Index — Distributed LAN LLM MVP

> Compiled 2026-09-13 by 5 parallel research crew mates. Full reports in this folder. Start here.

## Crew roster and reports

| # | Mate | Report | One-line intel |
|---|------|--------|----------------|
| 1 | Similar-projects scout | `01-similar-projects.md` | Pipeline-over-LAN is right; copy Ollama's API/packaging, exo's placement, Petals' routing math — not their transports |
| 2 | Rust stack researcher | `02-rust-stack.md` | Full crate list pinned (`quinn 0.11`, `llama-cpp-2 0.1.156`…); **layer-range execution is NOT in llama.cpp's public API — biggest technical risk** |
| 3 | Qwen3-0.6B analyst | `03-qwen3-model.md` | Exact arch numbers, download links, KV math verified (112 KiB/tok); **recommended shard split 9/10/9, not 16/8/4** |
| 4 | Android worker researcher | `04-android-worker.md` | JNI path proven, tok/s estimates 12–25/s; **QUIC-on-Android too costly for MVP — ship TCP+TLS behind a transport interface** |
| 5 | Pipeline protocol researcher | `05-pipeline-protocol.md` | **Replace per-token 2PC COMMIT with optimistic primary-backup + piggyback**; block=16, recompute-preempt, SARATHI batching |

## Design changes adopted from research (master plan deltas)

1. **Shard split → 9/10/9 balanced** (`0–8 / 9–18 / 19–27`). The doc's 16/8/4 is ~4:2:1 imbalanced; keep 16/8/4 only as a documented heterogeneity fit.
2. **COMMIT protocol → optimistic.** Stages append `KV_TENTATIVE` without waiting; coordinator piggybacks `COMMIT(pos-1)` on next dispatch; abort = `TRUNCATE(pos)`. No extra RTT in steady state (was: 2PC-style wait-all-ack per token).
3. **Phase 3 must open with a layer-execution spike.** `llama-cpp-2` exposes no "run layers i–j" API. Ranked options: (a) llama.cpp RPC-backend sidecars, (b) custom JNI/C++ shard forwarders owning the graph, (c) fork of sys crate (unstable), (d) candle (slower). Data-parallel replicas + routing remain the fallback that still ships value.
4. **Android transport: TCP+TLS first, QUIC later** behind an `InferenceTransport` interface. Cronet is HTTP-only, quiche-JNI stale, libmsquic JNI surface is weeks of work.
5. **KV paging: block 16, recompute-preempt (never swap over LAN)**, promote to 32 only if p50>1K and preemption<5%.
6. **Qwen3 runtime defaults:** thinking mode on (`T=0.6/TopP=0.95/TopK=20/MinP=0`), `presence_penalty≈1.5`, context default 32768, `--jinja` flag; pin one GGUF publisher (unsloth Q4_K_M 397 MB recommended); validate Q4_K_M on ARM builds (repack-crash regression window exists).
7. **Android FGS type = `connectedDevice`** (not `dataSync` — capped 6h/24h on API 35+), partial wake lock only during compute, per-OEM onboarding screens required.
8. **Coordinator SLO knobs (only 3):** `max_batched_tokens`, `prefill_chunk` (256–512), `checkpoint_K` (64–128). Autoscale rules in report 05.
9. **Cross-ISA contract is token-level, not logit-level.** Acceptance: `max|Δlogit|<0.5` + top-1 >99.5% at batch-1/temp-0; never chase bitwise identity ARM==x86.
10. **Keep `P<=3` stages on Wi-Fi** (`<=5` wired). Halving stages beats fatter pipes/compression for decode latency.
