# 01 — Similar Projects Scout Report

**Scope:** pipeline-parallel, GGUF shards, QUIC+mTLS, Windows coordinator + Android workers, Ollama-like UX, Qwen3-0.6B first. Verified via web search + README/docs fetch, Sep 2026.

## 1. Ollama — `github.com/ollama/ollama`

- Single-node, Ollama-UX standard: `pull/run/ps/show`, local daemon + CLI. Go server + bundled `llama.cpp` runner.
- Stack: Go (`server/`, `api/`, `cmd/`), embedded `llama.cpp` via `fs/ggml/`, Go templates for chat.
- Parallelism: none distributed. Local multi-GPU only via `llama.cpp` tensor offload.
- Model format: GGUF first-class; imports SafeTensors/HF via conversion; detects `tokenizer.chat_template` from GGUF KV.
- Packaging/API: OCI-inspired `~/.ollama/models/manifests/...` + content-addressed `blobs/sha256:*`, layer media-types. `Modelfile` DSL. REST `/api/generate`, `/api/chat`, `/api/pull|push|tags|show|create`, `/api/blobs/:digest`, plus OpenAI-compat `/v1/chat/completions`. Parallel 16-part blob download with resume.
- Transport: plain HTTP on `127.0.0.1:11434`, no auth by default. Docker-like registry protocol (manifest fetch → blob fetch → SHA verify).
- License: MIT.
- **STEAL:** API shape (`/api/*` + OpenAI `/v1/*`), SSE streaming, `Modelfile` idea, content-addressed blob store + manifest-last push, `show` inspectability.
- **AVOID:** no distribution, no auth, no Windows/Android worker story. Don't copy unauthenticated defaults.

URLs: `https://github.com/ollama/ollama`, `https://docs.ollama.com/api`

## 2. llama.cpp + llama-server — `github.com/ggml-org/llama.cpp`

- Foundational C/C++ engine (`ggml` + model code) + `llama-server` HTTP daemon. Rolling `bNNNNN` builds, no semver.
- Parallelism: local `-ngl 99` + `--tensor-split`; distributed `--rpc host:port,...` + `rpc-server` workers forwarding each `ggml` graph op over TCP; scheduler spreads layers/tensors by advertised memory. New RDMA transport with TCP fallback.
- Model format: GGUF native. Sharded GGUF (`-00001-of-00009.gguf`) auto-loads siblings.
- Transport: `llama-server` HTTP OpenAI-compat (`--api-key` optional, binds loopback by default). RPC = custom unauthenticated TCP (+optional RDMA).
- License: MIT.
- **STEAL:** GGUF loader/sampler/quant kernels, `llama-server` wire shape, `-np` parallel slots, `--tensor-split` weighting intuition, `gguf-split`.
- **AVOID:** raw RPC for product — manual IP lists, no discovery/health/rebalance, version-mismatch hangs, open unauth TCP, per-op round-trips. Single-machine-that-fits always beats RPC cluster on speed.

URLs: `https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md`

## 3. exo — `github.com/exo-explore/exo`

- Zero-config home-cluster: pool Macs/Linux/iPhones into one inference endpoint. Closest UX analogue to "LAN distributed Ollama."
- Stack: Python orchestrator + MLX (Apple) / tinygrad (CUDA/CPU). Dashboard + chat UI on `:52415`.
- Parallelism: pipeline "ring memory-weighted partitioning" sized by device RAM, topology-aware; newer tensor-parallel claims + RDMA-over-Thunderbolt. Dynamic rebalance on churn.
- Model format: HF SafeTensors via MLX/tinygrad — **not GGUF**. Biggest divergence from our plan.
- Transport: UDP-multicast discovery (or Tailscale/manual IPs), P2P mesh (no master), TCP + RDMA-over-TB5. OpenAI/Claude/Ollama-compat API.
- License: Apache-2.0.
- **STEAL:** auto-discovery, memory-weighted placement, topology graph, `switch base-url and done` API compat, honest benchmark culture (2-Mac Thunderbolt can be *slower* than 1-Mac if model fits — cluster wins only when model doesn't fit).
- **AVOID:** P2P-no-coordinator (we want Windows coordinator); MLX-first = Apple-only fast path, no Windows/Android story; Wi-Fi clusters disappoint (need Ethernet/TB).

URLs: `https://github.com/exo-explore/exo`, `https://exolabs.net/`

## 4. Petals / hivemind — `github.com/bigscience-workshop/petals`

- BitTorrent-style volunteer swarm for 70B–405B models. Client holds embeddings+head, servers hold contiguous block spans.
- Stack: Python, PyTorch + Transformers, `hivemind` DHT (Kademlia + libp2p).
- Parallelism: pipeline over WAN. Servers announce spans to DHT; client builds (server,block) graph with RTT/2 + serialization + KV-pressure edges, Dijkstra min-latency chain. Client caches activations for failover reroute.
- Model format: SafeTensors only. Pinned old transformers.
- License: MIT. Paper: Borzunov et al, NeurIPS'23 `arxiv:2312.08361`. **Dormant since Sep 2024 — reference design, not dependency.**
- **STEAL:** Dijkstra-over-(server,block) routing with cache-pressure term, fail-fast on coverage gaps, activation-replay failover, trusted-private-swarm pattern.
- **AVOID:** public DHT + Internet threat model (servers see activations), churn stalls, per-token full-chain WAN latency, greedy placement under-utilizes heterogeneous nodes. Total overkill for trusted LAN — use mDNS + coordinator, not Kademlia.

URLs: `https://github.com/bigscience-workshop/petals`, `https://petals.dev/`

## 5. vLLM — `github.com/vllm-project/vllm`

- Production throughput engine for NVIDIA datacenters. Python scheduler + custom CUDA kernels, Ray multi-node, NCCL collectives.
- Parallelism: tensor-parallel (within node), pipeline-parallel (`--pipeline-parallel-size`, microbatched), data/expert-parallel for MoE. Guidance: TP=size-of-node, PP=number-of-nodes.
- Model format: SafeTensors; GGUF only via experimental out-of-tree plugin.
- **PagedAttention:** KV split into fixed-token blocks, non-contiguous physical blocks via block table (OS VM analogy), on-demand allocation (<4% waste vs ~60%+ fragmented), CoW sharing for parallel sampling (up to 55% mem save, 2.2x throughput).
- License: Apache-2.0.
- **STEAL:** paged KV-block manager concept, continuous batching, microbatch pipeline scheduling, uneven-split support, prefix caching.
- **AVOID:** CUDA+Ray+NCCL unusable for Windows+Android LAN; no GGUF story; operational weight far beyond 0.6B MVP.

URLs: `https://docs.vllm.ai/en/stable/design/paged_attention/`, `https://vllm.ai/blog/2023-06-20-vllm`

## 6. Model-serving UX pack

- **HuggingFace TGI** (`github.com/huggingface/text-generation-inference`): Rust router + Python servers, gRPC, SSE streaming, Prometheus/OTel. SafeTensors only, CUDA-only, **now in maintenance mode** (HF points to vLLM/SGLang/llama.cpp). Steal SSE shape + router/launcher split; avoid as dependency.
- **llama-box** (`github.com/gpustack/llama-box`): pure-API server on `llama.cpp`, OpenAI-compat, GGUF, exemplary prebuilt backend matrix. Single-node, **ARCHIVED**. Steal build matrix; avoid building on archived fork.
- **LocalAI** (`github.com/mudler/LocalAI`): Go core + 60+ gRPC backends, `local-ai run huggingface://...` gallery UX, on-demand backend pull, multi-API shim. MIT. Steal plugin pattern + on-demand pull; avoid scope sprawl.
- **Jan** (`github.com/janhq/jan`): Tauri desktop + bundled `llama.cpp` **router** (one server, `load/unload` per model, per-model overrides). Apache-2.0 (since Sep 2025). Single-node. Steal router-preset multi-model pattern; avoid desktop-app weight.
- **GPT4All** (`github.com/nomic-ai/gpt4all`): Qt desktop + Python client over `llama.cpp`, 2-minute gallery onboarding, offline-first. MIT. Steal onboarding; avoid desktop-first server.

## 7. GGUF shard-splitting tooling

- In-tree `tools/gguf-split` (`llama-gguf-split`): `--split`/`--merge`, `--split-max-tensors 128` or `--split-max-size`, `-00001-of-0000N` naming, each shard valid GGUF with `general.split_count` KV, sibling auto-discovery. MIT.
- Exists for **distribution** (file-size limits, interrupted transfers) — splits by tensor-count/size, **not** layer-contiguous pipeline stages. Cannot point worker A at shard 3 and get layers 12–18.
- **STEAL:** naming, sibling auto-discovery, merge flow, `--dry-run` planner; use for size-capped transfer chunking.
- **AVOID:** mistaking tensor-shards for pipeline shards. We need our own **layer-contiguous splitter** (whole GGUF per stage with compute restricted to range, or repacked per-stage GGUFs + manifest mapping layer-range → worker).

URLs: `https://github.com/ggml-org/llama.cpp/blob/master/tools/gguf-split/README.md`

## 8. quinn (Rust QUIC) — `github.com/quinn-rs/quinn`

- Pure-Rust async IETF QUIC (tokio, rustls+ring). Stable Rust, Windows supported. One UDP socket per endpoint, simultaneous client/server, ordered+unordered streams, datagrams.
- mTLS fit: rustls-native — custom verifiers, `rcgen` self-signed CA, persist-and-reuse certs for trust-on-first-use.
- Production proof: **iroh** ran Quinn on hundreds of thousands of devices, then forked (`iroh-quinn`/`noq`) only for multipath/NAT-traversal — upstream Quinn stays the default for plain QUIC. Avoid `s2n-quic` (metrics-heavy) and `quiche` (BoringSSL C dep).
- Ops notes: bump `SO_SNDBUF/SO_RCVBUF`, keep rustls versions aligned, `platform-verifier` + `qlog` useful.
- License: MIT + Apache-2.0.

URLs: `https://github.com/quinn-rs/quinn`, `https://www.iroh.computer/blog/why-we-forked-quinn`

## Top-10 lessons for our build

1. **Pipeline, not tensor, for LAN heterogeneity.** Tensor-parallel needs NVLink-class bandwidth; pipeline (one activation hop per stage per token) is the only sane LAN choice.
2. **Copy Ollama's API + packaging, not its engine.** `/v1/chat/completions` + `/api/*`, SSE, `pull/run/ps`, content-addressed blobs + manifest-last.
3. **Keep GGUF whole per stage; write our own layer-splitter.** Reuse `-00001-of-0000N` naming only for download chunking.
4. **Learn from llama.cpp RPC, don't adopt it.** Unauth TCP, manual lists, no discovery/failover. Our QUIC+mTLS + coordinator placement + heartbeats is the fix.
5. **Steal exo's placement, invert its topology.** Memory-weighted latency-aware assignment + mDNS discovery = right; P2P-no-master = wrong for us.
6. **Steal Petals' routing math, skip its DHT.** Score chains by compute + RTT/2 + KV-pressure; LAN mDNS + pinned CA instead of Kademlia.
7. **Take vLLM's KV insight at small scale.** Block-paged KV + prefix caching + continuous microbatching; nail correct single-stream SSE first, keep scheduler batch-ready.
8. **quinn+rustls is the right transport.** Coordinator CA, persisted worker certs, TOFU + fingerprint log, tuned UDP buffers, one connection per worker multiplexing control/activations/bulk.
9. **Use Jan's router pattern for multi-model.** One coordinator, `[*]` defaults + per-model overrides, on-demand load/unload.
10. **Scope to Windows+Android.** No Ray/NCCL/CUDA-graphs, no MLX, no PyTorch/DHT, no desktop shell. MVP: coordinator (Windows QUIC server, scheduler, Ollama-compat API) + Android workers (QUIC client, CPU GGUF stage exec), Qwen3-0.6B Q4_K_M pipeline, `pull → run → chat` over SSE.
