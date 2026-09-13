# DLLM Mesh — Android (Phase 0 scaffold)

Native Kotlin + Compose chat client / future worker. Package `com.dllm.mesh`,
`minSdk 28`, `targetSdk 34`, version `0.1.0-phase0`.

## What works in Phase 0

- App skeleton with bottom nav: **Chat** + **Devices**.
- Chat screen: message list, input + Send, attach-to-`session_id` field.
  Streams tokens over SSE (`SseClient`) with `Last-Event-ID` resume;
  `session_id` + `last_event_id` persist in DataStore.
- Devices screen: coordinator address field, mDNS discovery list
  (`_dllm._tcp.` via `NsdDiscovery`), device list with approve/revoke
  hitting `GET /v1/devices`, `POST /v1/devices/{id}/approve`,
  `DELETE /v1/devices/{id}`, worker role toggle driving `WorkerService`.
- `WorkerService` starts as a `connectedDevice` foreground service with a
  silent channel (survival plumbing only — no compute yet).

## What is stubbed

- **Worker compute**: `runShardCompute()` / `connect()` are lock-lifecycle
  stubs. Real JNI shard execution lands in Phase 4.
- **QUIC**: transport is OkHttp TCP+TLS/SSE behind `SseClient`, per research
  decision (QUIC later behind the same shape).
- **Pairing crypto**: QR show/scan moves a JSON join payload
  (`url`/`node_id`/`fingerprint`); QR bitmap rendering, verify codes,
  Keystore signing, and mTLS allow-list land in Phase 3. Show screen is a
  selectable-text fallback (no zxing dependency).
- **API shape**: `contracts/openapi.yaml` + `contracts/pairing.md` were
  absent at scaffold time. Endpoint paths above are assumptions from
  `docs/MASTER_PLAN.md` — reconcile with the contracts mate before Phase 1.

## Build

Requires JDK 17 to run Gradle (see setup script warning about Java 25).

```powershell
.\setup-android.ps1            # installs SDK 34, writes local.properties
.\gradlew assembleDebug        # or: gradle assembleDebug
```

APK: `app/build/outputs/apk/debug/app-debug.apk`.
