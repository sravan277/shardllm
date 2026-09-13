# 04 — Android Chat + Worker Research (target `minSdk 28`, Kotlin + Compose, `arm64-v8a`; verified Sep 2026)

## 1. llama.cpp on Android — official path only

Template: `examples/llama.android` (`llama-android.cpp` JNI + `LLamaAndroid.kt` facade + `CMakeLists.txt`). Docs: `docs/android.md`, `docs/build.md`.

- Native build: `CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/.../android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF`.
- JNI bridge: thin `extern "C" JNIEXPORT` funcs returning `jlong` handles (model → context → batch → `llama_decode` loop → `llama_sampler_*`). Kotlin facade `load(path)/send(message): Flow<String>/bench()` on a dedicated thread, `System.loadLibrary("llama-android")`.
- GGUF load: `use_mmap=true` (default) — OS pages weights; 800MB–1.1GB Q4 shows ~800MB virtual but ~100–300MB resident + KV. App-private path (`filesDir`/`getExternalFilesDir`); **mmap can't work on a `ContentResolver` Uri — copy first.** Context size dominates RAM: 0.6B/2048ctx ≈ 64MB KV.
- ABIs: **`arm64-v8a` only for MVP** (+`x86_64` for emulator). No `armeabi-v7a` (no DOTPROD/i8mm, 2–3× slower).
- Realistic 0.6B-class tok/s: **mid phone 12–25 t/s gen, 200–800ms TTFT @ctx2048** (Cortex-X3 Q8_0: 17–19 t/s; SD 7s Gen3 1B Q4: 8–12 t/s; flagship X300 1.7B Q4: 34 t/s). **Q4_K_M beats Q8_0 on ARM.** Compile `-march=armv8.7a` (or `armv8.6-a+dotprod+i8mm` + `GGML_CPU_KLEIDIAI=ON`). CPU-only MVP; OpenCL/Vulkan/QNN marginal for the effort.
- Pitfalls: NDK OpenMP unsupported (force OFF); `llamafile` OFF; `libstdc++.so.6` error = wrong STL (use `c++_shared`); thermal throttling halves sustained tok/s — benchmark warm.

## 2. Layer-RANGE via JNI — feasible, NOT via `llama.h` (C++ work; separate `worker` module, not MVP chat)

No `load_layers(i,j)` API; `llama_context` assumes full model. Options ranked:
1. **MVP-cheap: RPC backend as-is.** Phone as `rpc-server` (TCP :50052); coordinator runs `llama-server --rpc <phone-ip>`; layer split via `--tensor-split`. Zero custom math. Downsides: TCP not QUIC, per-op LAN round-trips, no phone→phone chaining without host relay.
2. **True pipeline shard:** custom JNI — mmap full file, create `ggml` tensors for `[L_START,L_END)` only; `nativeForwardShard(in: DirectByteBuffer): DirectByteBuffer`; own per-layer KV on-device; exact hidden-dim/dtype match (fp16/fp32). ~300–600 LOC C++ against `ggml.h`, bypassing `llama_decode`. **Pin llama.cpp commit on both ends — wire format changes break silently.**
3. **Reject:** `llama_decode` on a "partial model" — crashes on missing tensors.

Note: activations are large (~2048 tok × 1024 dim × 2B ≈ 4MB per forward per range — LAN-OK on streams, not datagrams). Pin big cores; handle prefill vs decode shapes; RPC has no auth — wrap in TLS+HMAC.

## 3. ForegroundService + WakeLock survival (only Play-legal "stay alive")

Pattern: **`ForegroundService (connectedDevice)`** + `PARTIAL_WAKE_LOCK` only during compute + `WifiLock` during QUIC + optional `MulticastLock` during discovery + battery-exemption request.

Manifest (API 34+ mandatory — type in manifest AND `startForeground(..., CONNECTED_DEVICE)`, else crash):
`FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_CONNECTED_DEVICE`, `WAKE_LOCK`, `ACCESS_NETWORK_STATE`, `CHANGE_WIFI_MULTICAST_STATE`, `POST_NOTIFICATIONS` (API 33+ runtime), `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS` (Play-policy-guarded). Service: `foregroundServiceType="connectedDevice"`, `exported="false"`.

- Why `connectedDevice` not `dataSync`: **`dataSync` on API 35+ capped at 6h/24h** (`onTimeout()` → must stop) and can't start from `BOOT_COMPLETED`. `connectedDevice` fits "LAN peer" with no timeout.
- `startForeground()` within ~10s; `START_STICKY`; silent `IMPORTANCE_MIN` channel. **FGS alone doesn't keep CPU awake when screen off — hold the wake lock during compute, release idle** (Play Vitals Mar 2026 penalizes >2h screen-off wake in >5% sessions). Doze still suspends network without user exemption.
- **OEM > AOSP:** Samsung Sleeping Apps (3d inactivity, resets on OTA), Xiaomi/HyperOS (kills + resets autostart on OTA, needs Autostart + No-restrictions prompt), Honor/Huawei PowerGenie (kills wakelocks >60min). No API bypass — per-OEM onboarding screens (`dontkillmyapp.com` intents), re-prompt after OTA.

## 4. QUIC on Android — DECISION: TCP+TLS MVP, QUIC later behind `InferenceTransport`

Single-hop LAN gains nothing from QUIC worth the JNI risk. `InferenceTransport: sendActivations(): Flow<Chunk>` abstraction lets QUIC drop in later.

| Option | Verdict |
|---|---|
| Cronet (`cronet-embedded`) | HTTP/3-over-QUIC client only — no raw streams/server. Good only if protocol is HTTPS POST/SSE. |
| OkHttp alone | No stable raw QUIC. |
| quiche JNI (`quiche4j`) | Raw QUIC but stale vs upstream; ALPN mismatch with MsQuic is failure #1. |
| libmsquic.so on Android | Best Windows interop (same impl) — but large JNI surface, ~weeks. Later. |
| **TCP+TLS via OkHttp (MVP)** | Self-signed cert pinned via QR fingerprint + length-prefixed binary or SSE/WebSocket. Mature, debuggable, identical trust both ends. `usesCleartextTraffic="false"`. |

QUIC-later checklist: ALPN `llm-shard/1`, QUIC v1 only, datagrams both ends, QR fingerprint certs, IPv4+IPv6-link-local, Wi-Fi lock held.

## 5. Discovery / QR / SSE / identity

- **mDNS `NsdManager`:** `registerService`/`discoverServices("_llmshard._tcp")`/`resolveService`; Android 14+ `registerServiceInfoCallback`. MulticastLock during discovery only; save conflict-renamed service name; `CHANGE_WIFI_MULTICAST_STATE`; **QR fallback mandatory** (routers block mDNS across bands).
- **QR CameraX + ML Kit bundled** (`barcode-scanning:17.3.0`, 2.4MB, offline-capable): `LifecycleCameraController` + `MlKitAnalyzer`, `STRATEGY_KEEP_ONLY_LATEST`, close `ImageProxy`.
- **SSE OkHttp:** `okhttp-sse` `EventSourceListener` (`onOpen/onEvent/onFailure/onClosed`), `Accept: text/event-stream`, `readTimeout(0)` (default 10s kills idle streams!), `Last-Event-ID` retry; or LaunchDarkly `BackgroundEventSource` (auto-reconnect); or hand-rolled `charStream` → `Flow`. Never `lifecycleScope` for worker streams.
- **Identity:** `AndroidKeystore` EC/RSA (alias in DataStore, never plaintext keys) or encrypted DataStore 1.3+ + Tink; single DataStore instance per file.

## 6. Wi-Fi-only / battery / storage / scoped storage

- Gate: `TRANSPORT_WIFI + NET_CAPABILITY_INTERNET + VALIDATED (+ NOT_METERED optional)`; continuous `registerNetworkCallback`; downloads via WorkManager `UNMETERED + RequiresBatteryNotLow + RequiresStorageNotLow`.
- Guards: `isCharging`/`BATTERY_PROPERTY_CAPACITY` (<20% refuse shard work); `StorageManager.getAllocatableBytes()` or `ACTION_MANAGE_STORAGE`.
- Shards at `filesDir/models/<sha256>.gguf` (internal, encrypted API29+, mmap-friendly) or `getExternalFilesDir/models/` if tight. Never MediaStore/Downloads. Copy SAF Uri → private file → SHA-256 verify → mmap.

## Minimal Gradle deps (MVP) + `.so` build sketch

`compose-bom 2026.01.00`, `core-ktx`, `lifecycle-runtime-ktx`, `okhttp + okhttp-sse + logging-interceptor 4.12.0`, `camera-*:1.6.2`, `mlkit barcode-scanning 17.3.0`, `datastore-preferences 1.2.0`, `work-runtime-ktx 2.10.0`; `minSdk 28`, ABIs `arm64-v8a+x86_64`, `c++_shared`, CMake 3.22.1. (WorkManager 2.9+: override manifest `SystemForegroundService` FGS type or `startForeground(DATA_SYNC)` throws.)

Build: pin llama.cpp submodule tag on both ends → `CMakeLists.txt` (`LLAMA_NATIVE OFF, GGML_OPENMP OFF, GGML_LLAMAFILE OFF, LLAMA_CURL/EXAMPLES/TESTS OFF`; `-march=armv8.7a -O3`) → ~100 LOC `shard_jni.cpp` (backend_init/load/new_context/forward_shard/free) → host cross-compile sanity → Gradle sync → verify `lib/arm64-v8a/libshard_jni.so` in APK → on-device run 0.6B Q4_K_M `n_ctx 2048` → then FGS + NsdManager + OkHttp layers.
