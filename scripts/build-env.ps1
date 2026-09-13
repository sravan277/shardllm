# Phase 1+ build environment (Windows, no admin, MinGW UCRT).
# Dot-source from the repo root before cargo:  . .\scripts\build-env.ps1
# Persists the env recipe from the llama-cpp-2 MinGW bring-up (see docs/SETUP.md).
# NOTE: never set $ErrorActionPreference='Stop' here — with 2>&1 redirection it
# turns cargo's stderr progress lines into terminating errors and kills the build.
$ErrorActionPreference = "Continue"

$WinLibs = "C:\Users\shiva\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe"
$MinGWBin = "$WinLibs\mingw64\bin"
$CmakeBin = Get-ChildItem "C:\Users\shiva\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe" -Recurse -Directory -Filter "bin" | Select-Object -ExpandProperty FullName -First 1
$NinjaExe = Get-ChildItem "C:\Users\shiva\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe" -Recurse -Filter "ninja.exe" | Select-Object -ExpandProperty FullName -First 1
$NinjaDir = Split-Path $NinjaExe -Parent

foreach ($d in @("$env:USERPROFILE\.cargo\bin", $MinGWBin, $CmakeBin, $NinjaDir)) {
    if ($env:PATH -notlike "*$d*") { $env:PATH = "$d;" + $env:PATH }
}

$env:LIBCLANG_PATH = "C:\Users\shiva\AppData\Local\Programs\Python\Python314\Lib\site-packages\clang\native"
$env:CMAKE_GENERATOR = "Ninja"
$env:CFLAGS = "-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00"
$env:CXXFLAGS = "-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00"

# bindgen needs MinGW headers with forward slashes (its shlex mangles backslashes),
# including GCC's freestanding headers (stdbool.h etc.) under lib/gcc/<ver>/include.
$wl = $WinLibs -replace '\\', '/'
$gccInc = Get-ChildItem "$WinLibs\mingw64\lib\gcc\x86_64-w64-mingw32" -Directory |
    Sort-Object Name -Descending | Select-Object -ExpandProperty FullName -First 1
$incs = @("$wl/mingw64/include", "$wl/mingw64/x86_64-w64-mingw32/include") | Where-Object { Test-Path ($_ -replace '/', '\') }
if ($gccInc) { $incs += (($gccInc + "\include") -replace '\\', '/') }
$env:BINDGEN_EXTRA_CLANG_ARGS = (($incs | ForEach-Object { "-I$_" }) + @("-D_WIN32_WINNT=0x0A00")) -join ' '

# llama-cpp-sys-2 only links advapi32 + llama-common-base.a for MSVC; MinGW needs them forced.
$CommonBase = Get-ChildItem "target" -Recurse -Filter "libllama-common-base.a" 2>$null |
    Sort-Object LastWriteTime -Descending | Select-Object -ExpandProperty FullName -First 1
if ($CommonBase) {
    $env:RUSTFLAGS = "-l advapi32 -C link-args=$CommonBase"
}

Write-Output "build env ready (common-base: $CommonBase)"
