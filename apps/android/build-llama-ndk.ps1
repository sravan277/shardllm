#Requires -Version 5.1
<#
.SYNOPSIS
  Reproducible CPU-only llama.cpp build for arm64-v8a (NDK clang toolchain).

.DESCRIPTION
  Phase 4 worker ships a JNI stub (libdllm_worker.so) so assembleDebug stays
  green. This script produces the REAL static libs (llama/ggml/common) that
  Phase 5 links behind the same LlamaBridge API — no binaries in git, just
  re-run this script user-space (no admin) to reproduce them.

  Pinned: llama.cpp tag b7418 (verified release, ggml-org/llama.cpp, Dec 2025;
  satisfies the "b7416+" requirement), NDK r26d (26.3.11579264),
  ANDROID_ABI=arm64-v8a, ANDROID_PLATFORM=android-28, no OpenMP/CUDA/Vulkan.

  Layout:
    app/src/main/cpp/third_party/llama.cpp/                 <- clone (not committed)
    app/src/main/cpp/third_party/llama.cpp/build-android-arm64/ <- .a outputs

  Prereqs (user-space, no admin):
    - Android SDK with NDK + CMake:
        & "$env:LOCALAPPDATA\Android\Sdk\cmdline-tools\latest\bin\sdkmanager.bat" `
            --install "ndk;26.3.11579264" "cmake;3.22.1"
    - git.exe (preferred) OR curl.exe fallback to the release tarball.
    - cmake + ninja: taken from SDK cmake package; system PATH as fallback.

  Phase 5 link step: set DLLM_WITH_LLAMA=ON in
  app/src/main/cpp/CMakeLists.txt and point LLAMA_ANDROID_DIR at
  third_party/llama.cpp/build-android-arm64.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build-llama-ndk.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build-llama-ndk.ps1 -LlamaTag b7418
#>
param(
    [string]$LlamaTag = "b7418",
    [string]$NdkVersion = "26.3.11579264",
    [string]$Abi = "arm64-v8a",
    [string]$Platform = "android-28",
    [string]$CmakeVersion = "3.22.1"
)

$ErrorActionPreference = "Stop"

$RepoAndroid = $PSScriptRoot
if ([string]::IsNullOrEmpty($RepoAndroid)) { $RepoAndroid = (Get-Location).Path }

function Find-AndroidSdk {
    if ($env:ANDROID_HOME -and (Test-Path -LiteralPath $env:ANDROID_HOME)) { return $env:ANDROID_HOME }
    if ($env:ANDROID_SDK_ROOT -and (Test-Path -LiteralPath $env:ANDROID_SDK_ROOT)) { return $env:ANDROID_SDK_ROOT }
    $cand = Join-Path $env:LOCALAPPDATA "Android\Sdk"
    if (Test-Path -LiteralPath $cand) { return $cand }
    throw "Android SDK not found. Set ANDROID_HOME or install to %LOCALAPPDATA%\Android\Sdk."
}

$Sdk = Find-AndroidSdk
Write-Host "SDK: $Sdk"
$NdkDir = Join-Path $Sdk ("ndk\" + $NdkVersion)
if (-not (Test-Path -LiteralPath $NdkDir)) {
    throw ("NDK $NdkVersion missing at $NdkDir. Install user-space (no admin): " +
        "& `"$Sdk\cmdline-tools\latest\bin\sdkmanager.bat`" --install `"ndk;$NdkVersion`" `"cmake;$CmakeVersion`"")
}
$Toolchain = Join-Path $NdkDir "build\cmake\android.toolchain.cmake"
if (-not (Test-Path -LiteralPath $Toolchain)) { throw "NDK toolchain file missing: $Toolchain" }
Write-Host "NDK: $NdkDir"

$CmakeExe = Join-Path $Sdk ("cmake\" + $CmakeVersion + "\bin\cmake.exe")
if (-not (Test-Path -LiteralPath $CmakeExe)) {
    $cmd = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($cmd) { $CmakeExe = $cmd.Source } else { throw "cmake not found. Install: sdkmanager --install `"cmake;$CmakeVersion`"" }
}
$NinjaExe = Join-Path (Split-Path -Parent $CmakeExe) "ninja.exe"
if (-not (Test-Path -LiteralPath $NinjaExe)) {
    $ncmd = Get-Command ninja.exe -ErrorAction SilentlyContinue
    if ($ncmd) { $NinjaExe = $ncmd.Source } else { Write-Warning "ninja.exe not found next to cmake; build may fail. Install `"cmake;$CmakeVersion`" via sdkmanager." }
}
Write-Host "cmake: $CmakeExe"
& $CmakeExe --version | Select-Object -First 1

$ThirdParty = Join-Path $RepoAndroid "app\src\main\cpp\third_party"
$SrcDir = Join-Path $ThirdParty "llama.cpp"
$BuildDir = Join-Path $SrcDir "build-android-arm64"
New-Item -ItemType Directory -Path $ThirdParty -Force | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $SrcDir "CMakeLists.txt"))) {
    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($git) {
        Write-Host "Cloning llama.cpp tag $LlamaTag (ggml-org)..."
        & git.exe clone --depth 1 --branch $LlamaTag https://github.com/ggml-org/llama.cpp.git $SrcDir
        if (-not $?) {
            Write-Warning "ggml-org remote failed; trying legacy ggerganov remote..."
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

Write-Host "Configuring (ABI=$Abi, platform=$Platform, CPU-only)..."
$ConfigArgs = @(
    "-G", "Ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
    "-DANDROID_ABI=$Abi",
    "-DANDROID_PLATFORM=$Platform",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_SERVER=OFF",
    "-DLLAMA_CURL=OFF",
    "-DGGML_OPENMP=OFF",
    "-DGGML_VULKAN=OFF",
    "-DGGML_CUDA=OFF",
    "-S", $SrcDir,
    "-B", $BuildDir
)
& $CmakeExe @ConfigArgs
if (-not $?) { throw "cmake configure failed." }

Write-Host "Building static libs (this takes a while on first run)..."
& $CmakeExe --build $BuildDir --config Release
if (-not $?) { throw "cmake build failed." }

Write-Host "--- Outputs ($BuildDir) ---"
Get-ChildItem -LiteralPath $BuildDir -Recurse -Filter "*.a" | Select-Object FullName, @{n="MB";e={[math]::Round($_.Length/1MB,1)}} | Format-Table -AutoSize
Write-Host "Done. Phase 5: flip DLLM_WITH_LLAMA=ON in app/src/main/cpp/CMakeLists.txt with LLAMA_ANDROID_DIR=$BuildDir"
