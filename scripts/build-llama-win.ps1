#Requires -Version 5.1
<#
.SYNOPSIS
  Host llama.cpp (x86_64 Windows, MinGW) with GGML_RPC=ON for the native dllm_shim.

.DESCRIPTION
  `llama-cpp-2` (the crate the Rust coordinator links against) vendors a
  llama.cpp tree with NO `ggml/src/ggml-rpc/` directory, and its bindgen never
  includes `ggml-rpc.h`. Zero `ggml_rpc*` symbols are therefore reachable from
  Rust, so layers cannot be executed on another device. `native/` works around
  that by owning its own C++ translation unit (native/src/dllm_shim.cpp) which
  links llama.cpp + ggml-rpc and exposes a plain C ABI.

  This script produces that llama.cpp. It pins the SAME tag as the Android NDK
  build (b7418) so the coordinator, the APK worker and the shim all speak one
  model format and one graph layout; a version skew between them would surface
  as mysterious tensor-name mismatches rather than as a build error.

  The only option that differs from the Android script is `-DGGML_RPC=ON`.
  Every other flag mirrors apps/android/build-llama-ndk.ps1 so the two builds
  differ only in target.

  Layout:
    third_party/llama.cpp/                 <- clone (gitignored, not committed)
    third_party/llama.cpp/build-win/       <- .a outputs, incl. libggml-rpc.a

  Prereqs (user-space, no admin), see docs/SETUP.md:
    - cmake + ninja: `winget install --source winget --exact --id Kitware.CMake
      --scope user` and the same for `Ninja-build.Ninja`. Dot-sourcing
      `scripts/build-env.ps1` puts both on PATH.
    - git.exe (preferred) OR curl.exe + tar.exe (Windows 10+) for the fallback.
    - WinLibs MinGW on PATH for g++/ar/ranlib.

  Then:
    cmake -S native -B native/build-win -G Ninja
    cmake --build native/build-win

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\build-llama-win.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\build-llama-win.ps1 -LlamaTag b7418 -Jobs 8
#>
param(
    # Same pin as apps/android/build-llama-ndk.ps1. Change both together or not
    # at all: the coordinator, the APK worker and the shim must agree.
    [string]$LlamaTag = "b7418",
    # 0 = let ninja use every core.
    [int]$Jobs = 0
)

# NOTE: never set $ErrorActionPreference='Stop' in a script that redirects
# stderr; it turns compiler progress lines into terminating errors. Every step
# below checks $? / $LASTEXITCODE explicitly instead.
$ErrorActionPreference = "Continue"

$RepoRoot = $PSScriptRoot
if ([string]::IsNullOrEmpty($RepoRoot)) { $RepoRoot = (Get-Location).Path }
$RepoRoot = Split-Path -Parent $RepoRoot

# ---------------------------------------------------------------- toolchain ---
$CmakeCmd = Get-Command cmake.exe -ErrorAction SilentlyContinue
if (-not $CmakeCmd) {
    throw ("cmake not found on PATH. Install user-space (no admin): " +
           "winget install --source winget --exact --id Kitware.CMake --scope user")
}
$CmakeExe = $CmakeCmd.Source

$NinjaCmd = Get-Command ninja.exe -ErrorAction SilentlyContinue
if (-not $NinjaCmd) {
    throw ("ninja not found on PATH. Install user-space (no admin): " +
           "winget install --source winget --exact --id Ninja-build.Ninja --scope user")
}
$NinjaExe = $NinjaCmd.Source

# g++ must exist, or the configure step picks a compiler that cannot link.
$CompilerCmd = Get-Command g++.exe -ErrorAction SilentlyContinue
if (-not $CompilerCmd) {
    throw ("g++.exe not found on PATH. WinLibs MinGW provides it (see docs/SETUP.md / " +
           "ADR-012); run `. .\scripts\build-env.ps1` first.")
}
$CompilerExe = $CompilerCmd.Source

Write-Host "cmake:   $CmakeExe"
& $CmakeExe --version | Select-Object -First 1
Write-Host "ninja:   $NinjaExe"
& $NinjaExe --version
Write-Host "cxx:     $CompilerExe"
& $CompilerExe --version | Select-Object -First 1

$ThirdParty = Join-Path $RepoRoot "third_party"
$SrcDir = Join-Path $ThirdParty "llama.cpp"
$BuildDir = Join-Path $SrcDir "build-win"
New-Item -ItemType Directory -Path $ThirdParty -Force | Out-Null
Write-Host "src:     $SrcDir"
Write-Host "build:   $BuildDir"

# ------------------------------------------------------------------ clone ---
if (-not (Test-Path -LiteralPath (Join-Path $SrcDir "CMakeLists.txt"))) {
    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($git) {
        Write-Host "Cloning llama.cpp tag $LlamaTag (ggml-org)..."
        & git.exe clone --depth 1 --branch $LlamaTag https://github.com/ggml-org/llama.cpp.git $SrcDir
        if (-not $?) {
            Write-Warning "ggml-org remote failed; trying legacy ggerganov remote..."
            if (Test-Path -LiteralPath $SrcDir) { Remove-Item -LiteralPath $SrcDir -Recurse -Force }
            & git.exe clone --depth 1 --branch $LlamaTag https://github.com/ggerganov/llama.cpp.git $SrcDir
            if (-not $?) { throw "git clone failed for tag $LlamaTag on both remotes." }
        }
    } else {
        # No git: curl.exe tarball fallback (never Invoke-WebRequest; user-space only).
        $tgz = Join-Path $env:TEMP ("llama.cpp-" + $LlamaTag + ".tar.gz")
        Write-Host "git.exe missing; downloading tarball via curl.exe..."
        & curl.exe -L --fail -o $tgz ("https://github.com/ggml-org/llama.cpp/archive/refs/tags/" + $LlamaTag + ".tar.gz")
        if (-not $?) { throw "curl.exe download failed for tag $LlamaTag." }
        $tmp = Join-Path $env:TEMP ("llama.cpp-" + $LlamaTag)
        if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
        New-Item -ItemType Directory -Path $tmp -Force | Out-Null
        # tar ships with Windows 10; falls back to Expand-Archive for .zip only.
        & tar -xzf $tgz -C $tmp
        if (-not $?) { throw "tar extraction failed: $tgz" }
        $inner = Get-ChildItem -LiteralPath $tmp -Directory | Select-Object -First 1
        if ($inner) {
            if (Test-Path -LiteralPath $SrcDir) { Remove-Item -LiteralPath $SrcDir -Recurse -Force }
            Move-Item -LiteralPath $inner.FullName -Destination $SrcDir
        } else { throw "Unexpected tarball layout in $tmp." }
    }
} else {
    Write-Host "llama.cpp sources already present at $SrcDir; leaving as-is."
}

if (-not (Test-Path -LiteralPath (Join-Path $SrcDir "ggml\src\ggml-rpc\ggml-rpc.cpp"))) {
    throw ("Pinned tag $LlamaTag has no ggml/src/ggml-rpc/ggml-rpc.cpp under $SrcDir. " +
           "Without it GGML_RPC=ON is a no-op and the shim offloads nothing. " +
           "Pick a tag that still ships the RPC backend.")
}

# -------------------------------------------------------------- configure ---
# Same option set as the Android script, plus GGML_RPC=ON. GGML_RPC is the whole
# reason this script exists: it produces libggml-rpc.a, which is what makes
# "layers execute on another device" possible at all.
Write-Host "Configuring (x86_64 host, CPU-only, GGML_RPC=ON)..."
$ConfigArgs = @(
    "-G", "Ninja",
    "-DCMAKE_MAKE_PROGRAM=$NinjaExe",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DGGML_RPC=ON",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_SERVER=OFF",
    "-DLLAMA_BUILD_TOOLS=OFF",
    # common/ (sampling + chat + grammar + httplib) is ~10 MB of .a and nothing
    # in native/ links it; skipping it removes ~40% of first-build time.
    "-DLLAMA_BUILD_COMMON=OFF",
    "-DLLAMA_CURL=OFF",
    "-DGGML_OPENMP=OFF",
    "-DGGML_VULKAN=OFF",
    "-DGGML_CUDA=OFF",
    "-DGGML_NATIVE=OFF",
    "-S", $SrcDir,
    "-B", $BuildDir
)
& $CmakeExe @ConfigArgs
if (-not $?) { throw "cmake configure failed for $SrcDir -> $BuildDir. Scroll up for the actual error." }

# ------------------------------------------------------------------ build ---
Write-Host "Building static libs (10+ min on first run)..."
$BuildArgs = @("--build", $BuildDir, "--config", "Release")
if ($Jobs -gt 0) { $BuildArgs += @("--parallel", $Jobs) }
& $CmakeExe @BuildArgs
if (-not $?) { throw "cmake build failed. Scroll up for the actual compiler/linker error." }

# ----------------------------------------------------------------- verify ---
# Do not report success on the strength of the build's exit code alone; prove
# that the one artifact the whole design rests on actually exists.
#
# Note the glob: ggml/CMakeLists.txt sets CMAKE_STATIC_LIBRARY_PREFIX "" on
# win32-mingw, so on this toolchain the archive is `ggml-rpc.a`, not
# `libggml-rpc.a`. Match both so the check is not MinGW-specific.
# -Path (not -LiteralPath): -LiteralPath silently disables -Include filtering.
$RpcLib = Get-ChildItem -Path $BuildDir -Recurse -File -Include "ggml-rpc.a", "libggml-rpc.a" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $RpcLib) {
    throw ("Build reported success but no ggml-rpc.a / libggml-rpc.a was produced under " +
           "$BuildDir. GGML_RPC=ON did not take effect; re-run against a fresh -B directory.")
}
$LlamaLib = Get-ChildItem -Path $BuildDir -Recurse -File -Include "llama.a", "libllama.a" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $LlamaLib) {
    throw ("Build reported success but no libllama.a was produced under $BuildDir. " +
           "native/ cannot link the shim without it.")
}

Write-Host "--- Outputs ($BuildDir) ---"
Get-ChildItem -LiteralPath $BuildDir -Recurse -Filter "*.a" |
    Select-Object @{n="Lib";e={$_.FullName.Substring($BuildDir.Length+1)}}, Length,
                  @{n="MB";e={[math]::Round($_.Length/1MB,2)}} |
    Sort-Object Lib | Format-Table -AutoSize
Write-Host ("ggml-rpc: {0} ({1} bytes)" -f $RpcLib.FullName, $RpcLib.Length)
Write-Host ("llama:    {0} ({1} bytes)" -f $LlamaLib.FullName, $LlamaLib.Length)
Write-Host "Done. Next: cmake -S native -B native/build-win -G Ninja -DDLLM_SHIM_BUILD_TESTS=ON"
