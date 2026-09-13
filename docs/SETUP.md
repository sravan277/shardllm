# Setup (Windows dev machine — no admin required)

## Rust / exe

1. `curl.exe -sL -o %TEMP%\rustup-init.exe https://win.rustup.rs/x86_64` → run with `-y --default-toolchain none --profile minimal --no-modify-path`.
2. `%USERPROFILE%\.cargo\bin\rustup toolchain install stable-x86_64-pc-windows-gnu --profile minimal` → `rustup default stable-x86_64-pc-windows-gnu`.
3. Linker: `winget install -e --id zig.zig --source winget --scope user` (user-space). Repo `.cargo/config.toml` already points the gnu target at `zig cc -target x86_64-windows-gnu` for both linking and `cc`-crate C builds (rusqlite bundled, ring).
4. Build: `cargo build -p dllm` → `target\debug\dllm.exe`. Run: `dllm serve --port 8080`.
5. If you have admin + VS Build Tools, MSVC toolchain also works (`rustup toolchain install stable-x86_64-pc-windows-msvc`).

## Web

- Node 24+/npm. `cd apps/web && npm install && npm run dev` (dev, :5173). `npm run build` → `dist/` (served by `dllm serve`).
- No external fonts/CDNs (LAN-offline rule) — system type stack + local CSS only.

## Android / APK

1. Install JDK 17 (Temurin, user scope). Machine Java 25 cannot run Gradle 8.9 (max 22).
2. Run `apps\android\setup-android.ps1`: installs cmdline-tools → platform-tools, `platforms;android-34`, `build-tools;34.0.0` → writes `local.properties` → `gradlew assembleDebug`.
3. APK at `apps/android/app/build/outputs/apk/debug/app-debug.apk`. Point app at `http://<lan-ip>:8080` (cleartext needs LAN exception or Phase-1 HTTPS).

## Shell notes (this environment)

- `Invoke-WebRequest` fails here (NonInteractive) — scripts must use `curl.exe`.
- `winget` needs `--source winget` (msstore source prompts and fails headless).
- Don't `cd` in tool calls; use full paths / workdir param.
