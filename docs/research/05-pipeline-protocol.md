# 05 — Pipeline-Parallel Protocol Research (prior-art grounding; design under review in MASTER_PLAN §6–8)

## 1. Pipeline origins: GPipe / PipeDream → inference

- **GPipe** (Huang et al., NeurIPS'19): micro-batch `M` over `K` cells; bubble `(K-1)/(M+K-1)`, negligible when `M>=4K`; boundary-only activation comms. Re-materialization N/A to us (KV replaces activations).
- **PipeDream** (Harlap et al., SOSP'19): 1F1B steady state, weight stashing/vertical sync (no backward pass in inference — reject both), auto-partition from short single-GPU profiling balancing compute vs comm.
- Inference successors: **EdgePipe** (heterogeneous DP partition, 10.6x/11.9x on 16 edge devices), **DNNPipe/PipeEdge** (re-partition on churn), **ParetoPipe** (no single optimum; block-level profiling mandatory; RPC overhead can dominate), **Petals** (client caches inter-stage activations → replay failed stage `O(t)` not `O(n·t²)`), **DejaVu** (disaggregate prefill/decode), **SARATHI** (chunked-prefills + decode-maximal batching; bubbles −6.29x, E2E +1.91x).
- **Key transfer failure:** GPipe `M>=4K` does NOT apply to single-token decode (`M=1` in flight, forward-only, no B to interleave). Per-token fill/drain = `(P-1)·(t_stage + t_hop)`. Only fixes: concurrent sessions/micro-batches, chunked prefill (SARATHI), prefill/decode disaggregation, fewer stages.
- Adopt: GPipe cost-variance minimization + short-profiling DP partition; Petals activation cache for replay.

## 2. vLLM PagedAttention + continuous batching

- Paging analogy (tokens=bytes, blocks=pages, seqs=processes); non-contiguous blocks via block table; waste 60–80% → <4%; CoW sharing (2.2x throughput, 24x vs HF baseline).
- **Block size: start 16** (ShareGPT sweet spot 16–128, short-seq Alpaca 16–32; fragmentation <8 toks avg). 32 only if p50>1k and preemption<5%. On CPU/ARM table overhead negligible — fragmentation dominates.
- Block manager: `waiting` + `running` queues; admit subject to KV blocks + `max_num_batched_tokens` (2048) + `max_num_seqs` (128); decodes first, remainder for prefill chunks; **preempt → RECOMPUTE** (never swap; recompute ≤20% worse than swap, better for small blocks).
- Continuous batching = Orca iteration-level scheduling (36.9x vs FasterTransformer on GPT-3 175B); selective batching (batch matmuls/norms, per-seq attention).
- Adopt: per-stage local paged KV (block table, refcount, CoW prefix sharing); two-queue scheduler + token budget + chunked prefill; global coordinator budget mapped to slowest-stage ms. Reject: CUDA kernel assumptions; cross-stage shared blocks (stage-local paging, coordinate only `seq_len` + token IDs).

## 3. Commit per token: 2PC is overkill — ADOPTED: optimistic primary-backup

- No deployed pipeline-inference system does per-token 2PC/Paxos. Petals/DejaVu: **primary + replay** (client keeps per-server activation cache; on failure replace server, replay `O(t)` inputs). Async background replication + watermark; resume from last replicated `t`.
- Deterministic-sampling trap: `temperature=0` NOT bitwise deterministic under batching (batch-invariance: reduction order depends on co-tenants; 1000× greedy Qwen3-235B → 80 distinct completions). Fix = batch-invariant kernels at throughput cost; `seed` best-effort only.
- Decode is **RTT-bound** (16 KiB vector ≈ 0.13 ms xmit vs 10 ms hop); prefill bandwidth-bound. `tok/s ≈ tokens_committed / (stages·max_hop_ms)`. 2PC doubles trips per token. QUIC's win is HoL-avoidance (separate streams), not bandwidth.
- **Decision:** reject per-token 2PC for ≤5 trusted fail-stop LAN nodes (KV append deterministic given `(seq_id,pos,token_id)` — nothing to arbitrate). **Optimistic commit + truncate-rollback:** stages append tentatively, reply `COMPUTED(pos,hash)`; coordinator piggybacks `COMMIT(pos-1)` on next dispatch; abort = `TRUNCATE(pos)`. LWW safe because sampling happens exactly once at tail with pinned params.

## 4. KV checkpointing

- Naive full snapshot prohibitive (+113% prefill on 70B; SSD replication 47x worse than in-shadow parity). Working patterns: **DejaVuLib streaming** (each worker streams KV to `(x+1)%N` per token, async; watermark `(x,j,t)`), **KevlarFlow** block-wise background replication on separate stream (2.3% overhead, MTTR −20x, p99 −2.8x under failure), **GhostServe** erasure coding (overkill for N≤5 — use 1:1 replica, drop under pressure, fall back to re-prefill), **Concordia** AOF/delta log.
- Exact resume per stage needs: committed token IDs `[0..T]`, paged KV + block table + `seq_len`, sampler/RNG state + params version, `layer_range_id/weight_version`. NOT weights or inter-stage activations.
- Adopt: async ring replication (page = 16 toks granularity) + consolidated snapshot every **K=64–128** committed tokens (token IDs + KV pages + sampler state). Target <5% steady-state overhead.

## 5. Calibration at pairing (<2s budget)

- Partition from measured `e_i` (per-layer ms) + boundary bytes + device scale + pairwise bandwidth; throughput = min over stages of max(compute, comm) with overlap.
- ARM vs x86 non-linear in layer count AND phase (prefill compute-bound favors x86 AVX512; decode memory-bound narrows gap; int8 dot throughput differs 4x). **Store per-node `(ms/layer_decode, ms/layer_prefill_chunk, hop_ms)` vector; balance on decode, constrain on prefill. Never FLOPS or single scale factor.**
- Protocol: warmup 5 (clocks/allocator/QUIC handshake, discard) → decode probe (1-tok forward over candidate slice, median of 10) → prefill probe (256-tok chunk, median of 3) → activation-sized echo for `t_comm`. Re-run on join/leave or RTT shift >20%. Provision CPU for comm threads (+cores cut pipeline latency ~33% 2→4).

## 6. Fast path + speculative decoding (llama.cpp 2026 state)

- `llama-server --spec-type`: `draft-eagle3` (1-layer transformer on target hidden states; 2–3x claimed; needs trained draft per target family), `ngram-cache/map-k` (draftless, zero-config, small win on repetitive traffic).
- **Adopt: fast path first** (skip pipeline when model fits — avoids `(P-1)` hop tax; highest-ROI mitigation). Spec-decoding **only on fast path** (EAGLE-3 if draft exists, else ngram-cache). Distributed speculative verification multiplies round trips per draft block — deferred.

## 7. Cross-ISA numeric consistency

- ggml per-block quant; `vec_dot` differs per ISA. Reference tolerances: RMSE 0.002 std / 0.0075 2-bit; dot error 0.02 (0.04 low-bit). INT8 QDQ bit-identical across ARM cores (discrete grid); x86 breaks via PMADDUBSW saturation (no ARM analogue). Cascade: 1-bit wobble flips argmax only on near-ties, then diverges permanently.
- Methodology: pin same GGUF + same commit, `temp=0/top_k=1/seed/n_batch=1/threads pinned`; 200-tok continuation; metrics `max|Δlogit|`, `mean|Δ|`, top-1 agree %, tie-margin histogram vs same-device baseline. **Pass: `max Δ<0.5` AND top-1 >99.5% AND no flip where margin<Δ.** Flips at small margins → pin sampler to tail + log margins; never chase bit-identity.
- Contract: same-GGUF-everywhere + single sampler + margin logging; INT8/Q4_K_M preferred cross-ISA over fp16; checkpoint/replay at **token-ID level**, never logits; per-stage independent sampling rejected.

## 10 concrete protocol recommendations (adopted)

1. **Primary-backup + piggybacked commit** (no extra steady-state RTT). Abort = `TRUNCATE(pos)`.
2. **Sample once, at the tail**, versioned sampler params; log top-1 margin per token.
3. **Per-stage paged KV, block=16, recompute-preempt**; global `max_batched_tokens` + per-stage budgets; 32 only if p50>1k and preemption<5%.
4. **SARATHI-style scheduling:** decode-maximal batches (1 prefill chunk 256–512 + decodes); uniform-compute batches; disaggregate long prefills.
5. **Async ring replication + K=64–128 checkpoint**; recovery replays suffix from activation cache, recomputes failed range only. <5% overhead target.
6. **Pairing calibration (<2s):** warmup 5, split-phase timing, DP partition on decode, re-run on membership/RTT change.
7. **Fast path first; pipeline only when necessary.** `P≤3` on Wi-Fi, `≤5` wired.
8. **QUIC traffic classes + TCP fallback:** one conn per hop; streams `activation(prio)/control/bulk`; 8 MiB frame cap; ALPN gate; CID-pinned. Placement fixes RTT, not QUIC.
9. **Token-level cross-ISA contract** (acceptance per §7). Checkpoints compare token IDs + hashes.
10. **3 SLO knobs only:** `max_batched_tokens`, `prefill_chunk`, `checkpoint_K`. Rules: preemption>5% → shrink batch; p99 decode inflated by prefill → shrink chunk; replay cost > snapshot cost → shrink K.
