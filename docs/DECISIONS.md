# Decisions (ADRs — append, never rewrite history)

- **ADR-001 — gnu + zig linker, no MSVC.** No admin → no VS Build Tools. `stable-x86_64-pc-windows-gnu` + zig as linker/`CC` (`.cargo/config.toml`). Revisit if a crate needs MSVC-only asm.
- **ADR-002 — Balanced 9/10/9 default shards** (0–8/9–18/19–27). 16/8/4 kept only as heterogeneity fit. (research/03 §7)
- **ADR-003 — Optimistic commit, not per-token 2PC.** Piggyback `COMMIT(pos-1)`, abort via `TRUNCATE(pos)`. (research/05 §3)
- **ADR-004 — Single-device fast path before pipeline.** Pipeline pays `(P-1)` hop tax; 0.6B fits one node. (review §1)
- **ADR-005 — Android TCP+TLS first, QUIC later** behind `InferenceTransport`. (research/04 §4)
- **ADR-006 — KV block 16, recompute-preempt.** 32 only if p50>1K and preemption<5%. Checkpoint K=64–128. (research/05)
- **ADR-007 — Curated catalog only.** One pinned publisher per quant (unsloth). No arbitrary GGUF URLs in MVP.
- **ADR-008 — FGS type `connectedDevice`.** `dataSync` capped 6h/24h on API 35+. Wake lock only during compute. (research/04 §3)
- **ADR-009 — Token-level cross-ISA contract.** Accept `max|Δlogit|<0.5` + top-1 >99.5%; never chase bitwise ARM==x86.
- **ADR-010 — Layer-execution spike opens Phase 3.** llama.cpp public API has no layer-range run; options ranked RPC-sidecar > custom shard forwarder > sys-fork > candle. Data-parallel fallback ships value regardless.
- **ADR-011 — Minimal tooling policy.** 2 remote MCPs (context7, gh_grep) + reputable project-local skills only; no low-trust binaries; heavy tools enabled per-task. Rejections logged in `docs/MCP_SKILLS.md`.
- **ADR-012 — WinLibs MinGW UCRT over zig-as-linker.** Bare `zig` as cargo linker chokes on rustc's `-fno-use-linker-plugin`, and `windows-sys` needs real `dlltool.exe`. User-space WinLibs gives gcc/ld/dlltool/ar with no admin.
