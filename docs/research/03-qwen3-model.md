# 03 — Qwen3-0.6B Model Dossier (ground truth from `config.json`, verified 2026-09-13)

## 1. Architecture

| Parameter | Value |
|---|---|
| `num_hidden_layers` | **28** (indices 0–27) |
| `hidden_size` | **1024** |
| `intermediate_size` | **3072** (SwiGLU gate+up+down) |
| Q heads / KV heads / head_dim | **16 / 8 / 128** (GQA group 2; explicit head_dim, projs sized off it) |
| `vocab_size` | **151936** |
| `tie_word_embeddings` | **true** (single shared embed/lm_head) |
| Context | **32768** native (card); config max 40960; tokenizer claims 131072 (don't promise — belongs to 4B+ sizes) |
| RoPE θ / sliding window | 1,000,000 / **none — full causal attention ×28** |
| Norm / act / QK-Norm | RMSNorm ε=1e-6 / SwiGLU(silu) / **yes, per-head QK-Norm (new in Qwen3)** |
| Params | ~**596M** total / ~440M non-embedding |
| License | Apache-2.0 |

Qwen3 extras: **hybrid thinking/non-thinking in one model** (`enable_thinking` template flag, default True; `<think>` 151667/`</think>` 151668; `/think`, `/no_think` switches). Sampling: thinking `T=0.6/TopP=0.95/TopK=20/MinP=0`, non-thinking `T=0.7/TopP=0.8/TopK=20`; **never greedy**; `presence_penalty≈1.5` recommended esp. for quants. 100+ languages; tool-call template baked in.

## 2. GGUF downloads (pin ONE publisher — byte offsets differ)

| Repo | Q4_K_M | Q8_0 | F16/BF16 |
|---|---|---|---|
| `Qwen/Qwen3-0.6B-GGUF` (official) | — | `Qwen3-0.6B-Q8_0.gguf` **639 MB** | — |
| `unsloth/Qwen3-0.6B-GGUF` ✅ recommended | `Qwen3-0.6B-Q4_K_M.gguf` **397 MB** | `Qwen3-0.6B-Q8_0.gguf` **639 MB** | BF16 1.2 GB |
| `bartowski/Qwen_Qwen3-0.6B-GGUF` | Q4_K_M **484 MB** | Q8_0 **805 MB** | bf16 1.51 GB |
| `ggml-org/Qwen3-0.6B-GGUF` | Q4_K_M (Files tab) | Q8_0 (Files tab) | F16 |

Size deltas for the "same" quant come from imatrix calibration + embed/output kept at Q8_0/F16 (`*_L`/`*_XL`) vs quantized with rest. Keep shared embed at Q8_0/F16 (quality precedent).

Download: `huggingface-cli download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf --local-dir .` or `./llama-cli -hf unsloth/Qwen3-0.6B-GGUF:Q4_K_M`.

## 3. Tokenizer

Byte-level BPE, `Qwen2Tokenizer` class; GGUF self-contained (`tokenizer.ggml.tokens` + `.merges` + `qwen2` pre-tokenizer + Jinja `tokenizer.chat_template` + bos/eos/pad). Vocab 151,936. Turn format `<|im_start|>{role}\n{content}<|im_end|>\n`; no default system message. llama.cpp: pass `--jinja` so embedded template (with `enable_thinking`) is honored.
**Gotcha:** under `enable_thinking=false`, history assistant turns render *without* thinking tags while the generation prompt injects them — breaks KV prefix-cache reuse across turns (QwenLM/Qwen3#1826). Normalize history with the patched template if implementing cross-request caching.

## 4. KV-cache math — verified ✔

`bytes/token = 2 (K+V) × L × kv_heads × head_dim × bytes_per_elem` → 2×28×8×128×2 = **114,688 B = 112 KiB/token (FP16)**. 4K ctx = **448 MiB**. Q8 KV halves (56 KiB/tok, 224 MiB @4K). llama.cpp default KV is F16 unless `-ctk/-ctv q8_0/q4_0`.

Per-shard share (FP16): shard layers × 4,096 B/token.

## 5. llama.cpp support

Minimum **b5092** (`unknown model architecture: 'qwen3'` below that). 0.6B dense arch well supported incl. CI. Known issues: (1) **ARM64 + Q4_K repack crash** b7175–b7721 (#18392) — pin/validate ARM builds, x86_64 unaffected; (2) repetition disease — adopt official recipe (`--presence-penalty 1.5 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -fa -sm row --no-context-shift`).

## 6. Shared tensors

Tied embed `token_embd.weight` [151936×1024] = 155.58M params ≈ **296.75 MiB (F16)** / ~156–166 MB (Q8_0). `output_norm` ≈ 2–4 KiB. Shared file = embed + output_norm + metadata ≈ **~300 MB (F16) / ~150–170 MB (Q8_0)** / ~70–90 MB (Q4, not recommended).

## 7. Shard plan — ADOPTED: balanced 9/10/9 (replaces 16/8/4)

All 28 layers architecturally identical (same dims, full attention, no hybrids/MoE) — **layer count is a linear proxy for compute AND cache. CONFIRMED.** The doc's 16/8/4 is arithmetically valid but **~4:2:1 imbalanced** (A does 4× C's compute every token).

| Shard | Layers | Count | KV @4K FP16 | ≈Weight Q4_K_M |
|---|---|---|---|---|
| S0 | 0–8 | 9 | 144 MiB | ~80 MB |
| S1 | 9–18 | 10 | 160 MiB | ~89 MB |
| S2 | 19–27 | 9 | 144 MiB | ~80 MB |
| Shared | — | — | — | ~150–300 MB (Q8_0/F16) |

Imbalance ≤ ~11% (vs 300%). Middle stage takes the extra layer (S0/S2 carry embed-project/norm overhead). Boundaries on layer multiples only; wire format between stages fixed `[batch, seq, 1024]` (F16 ≈ 2 KiB/token/edge). Keep 16/8/4 only as documented heterogeneity fit with KV budgets 256/128/64 MiB @4K.

## Quick-reference numbers

Layers/hidden/FFN 28/1024/3072 · Q/KV/head 16/8/128 · vocab 151936 tied · ctx 32768 (40960 max) · RoPE 1M, no SWA · RMSNorm 1e-6, SwiGLU, QK-Norm · 596M/440M params · embed 155.58M ≈ 296.75 MiB F16 · KV 112 KiB/tok, 448 MiB @4K · Q4_K_M 397 MB (unsloth) · Q8_0 639 MB · llama.cpp ≥ b5092 · BPE/Qwen2Tokenizer/ChatML+think · presence-penalty 1.5.

Primary URLs: `huggingface.co/Qwen/Qwen3-0.6B` · `/Qwen/Qwen3-0.6B-GGUF` · `/unsloth/Qwen3-0.6B-GGUF` · `qwenlm.github.io/blog/qwen3` · `arxiv.org/abs/2505.09388` · `qwen.readthedocs.io/.../llama.cpp.html`
