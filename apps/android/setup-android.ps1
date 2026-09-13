#Requires -Version 5.1
<#
.SYNOPSIS
  Idempotent Android SDK bootstrap + debug build for DLLM Mesh.
.DESCRIPTION
  Downloads cmdline-tools to %LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest,
  accepts licenses, installs platform-tools / platforms;android-34 /
  build-tools;34.0.0, writes local.properties (sdk.dir), then runs
  assembleDebug. Safe to re-run. Uses full paths only (no cd).
.NOTES
  Gradle 8.9 cannot run on Java 25 — a JDK 17 is required. The script warns
  and prefers an installed JDK 17 when it finds one.
#>
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sdkRoot = Join-Path $env:LOCALAPPDATA "Android\Sdk"
$cmdToolsDir = Join-Path $sdkRoot "cmdline-tools\latest"
$cmdToolsZip = Join-Path $env:TEMP "android-cmdlinetools-win.zip"
$cmdToolsUrl = "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip"
$localProps = Join-Path $projectRoot "local.properties"

function Test-Jdk17 {
    $candidates = @(
        $env:JAVA_HOME,
        "C:\Program Files\Eclipse Adoptium\jdk-17",
        "C:\Program Files\Java\jdk-17",
        "C:\Program Files\Microsoft\jdk-17"
    ) | Where-Object { $_ -and (Test-Path -LiteralPath (Join-Path $_ "bin\java.exe")) }
    return $candidates | Select-Object -First 1
}

# --- 0. JDK check (Gradle 8.9 maxes out at Java 22; Java 25 will not work) ---
$jdk17 = Test-Jdk17
try {
    $javaVer = (& java -version 2>&1 | Out-String)
    Write-Host "Detected: $javaVer"
    if ($javaVer -match 'version "(\d+)') {
        $major = [int]$Matches[1]
        if ($major -gt 22) {
            if ($jdk17) {
                Write-Warning "Default java is $major; using JDK 17 at $jdk17 for this build."
                $env:JAVA_HOME = $jdk17
            } else {
                Write-Warning "Default java is $major but Gradle 8.9 needs <= 22. Install a JDK 17 and set JAVA_HOME."
            }
        }
    }
} catch {
    Write-Warning "No 'java' on PATH. Install JDK 17 before building."
}

# --- 1. cmdline-tools ---
if (-not (Test-Path -LiteralPath (Join-Path $cmdToolsDir "bin\sdkmanager.bat"))) {
    Write-Host "Downloading Android cmdline-tools..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $cmdToolsUrl -OutFile $cmdToolsZip -UseBasicParsing
    $stageDir = Join-Path $env:TEMP "android-cmdtools-stage"
    if (Test-Path -LiteralPath $stageDir) { Remove-Item -LiteralPath $stageDir -Recurse -Force }
    Expand-Archive -LiteralPath $cmdToolsZip -DestinationPath $stageDir -Force
    $inner = Join-Path $stageDir "cmdline-tools"
    if (Test-Path -LiteralPath $cmdToolsDir) { Remove-Item -LiteralPath $cmdToolsDir -Recurse -Force }
    New-Item -ItemType Directory -Path (Split-Path -Parent $cmdToolsDir) -Force | Out-Null
    # The zip contains cmdline-tools/ with the payload directly inside.
    Move-Item -LiteralPath $inner -Destination $cmdToolsDir -Force
    Remove-Item -LiteralPath $stageDir -Recurse -Force
    Remove-Item -LiteralPath $cmdToolsZip -Force
} else {
    Write-Host "cmdline-tools already present at $cmdToolsDir"
}

$sdkmanager = Join-Path $cmdToolsDir "bin\sdkmanager.bat"
if (-not (Test-Path -LiteralPath $sdkmanager)) { throw "sdkmanager not found at $sdkmanager" }

# --- 2. Licenses + packages (idempotent: sdkmanager skips installed pkgs) ---
Write-Host "Accepting licenses..."
& cmd /c "echo y | `"$sdkmanager`" --sdk_root=`"$sdkRoot`" --licenses" | Out-Null

Write-Host "Installing platform-tools, android-34, build-tools 34.0.0..."
& "$sdkmanager" --sdk_root="$sdkRoot" "platform-tools" "platforms;android-34" "build-tools;34.0.0"

# --- 3. local.properties ---
$escapedSdk = $sdkRoot -replace "\\", "\\"
Set-Content -LiteralPath $localProps -Value "sdk.dir=$escapedSdk" -Encoding Ascii
Write-Host "Wrote $localProps"

# --- 4. Build without changing directories ---
$gradlew = Join-Path $projectRoot "gradlew.bat"
if (Test-Path -LiteralPath $gradlew) {
    & "$gradlew" -p "$projectRoot" assembleDebug
} else {
    Write-Warning "gradlew.bat not found; falling back to system 'gradle'. Generate the wrapper with: gradle wrapper --gradle-version 8.9"
    & gradle -p "$projectRoot" assembleDebug
}
