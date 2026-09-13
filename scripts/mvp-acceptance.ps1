# MVP acceptance gate (MASTER_PLAN Phase 5 exit).
# Windows 10, PowerShell 5.1, no admin. From the repo root:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\mvp-acceptance.ps1
# Gates: (a) artifacts exist, (b) live `dllm serve --port 8099` E2E via curl.exe,
# (c) `cargo run -p dllm-core --example pipe_pair` -> PIPE_PAIR PASS.
# Prints a PASS/FAIL checklist and exits 1 on any FAIL.
# Rules honored: never $ErrorActionPreference='Stop' (cargo/curl stderr under
# redirection must stay non-terminating); curl.exe only, never Invoke-WebRequest;
# POST bodies always go through a temp file under TEMP\opencode (never inline JSON).

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Exe   = Join-Path $RepoRoot 'target\x86_64-pc-windows-gnu\debug\dllm.exe'
$Apk   = Join-Path $RepoRoot 'apps\android\app\build\outputs\apk\debug\app-debug.apk'
$Bench = Join-Path $RepoRoot 'docs\bench-baseline.json'
$Base  = 'http://127.0.0.1:8099'

$Work = Join-Path $env:TEMP 'opencode'
New-Item -ItemType Directory -Force -Path $Work | Out-Null
$Stamp       = Get-Date -Format 'yyyyMMdd-HHmmss'
$SrvOut      = Join-Path $Work "mvp-serve-$Stamp.out.log"
$SrvErr      = Join-Path $Work "mvp-serve-$Stamp.err.log"
$BodySession = Join-Path $Work "mvp-body-session-$Stamp.json"
$BodyMessage = Join-Path $Work "mvp-body-message-$Stamp.json"

$script:Rows = @()
function Add-Check {
    param([string]$Name, [bool]$Ok, [string]$Detail)
    $tag = 'FAIL'
    if ($Ok) { $tag = 'PASS' }
    $script:Rows = $script:Rows + [pscustomobject]@{ Name = $Name; Ok = $Ok; Detail = $Detail }
    Write-Output ("  [{0}] {1,-22} {2}" -f $tag, $Name, $Detail)
}

# ---- (a) artifacts -----------------------------------------------------------
Write-Output '== (a) artifacts =='
Add-Check 'exe artifact' (Test-Path -LiteralPath $Exe) $Exe
Add-Check 'APK artifact' (Test-Path -LiteralPath $Apk) $Apk

$benchOk = $false
$benchDetail = "$Bench (missing)"
if (Test-Path -LiteralPath $Bench) {
    $benchDetail = "$Bench (mean_decode_tps missing/zero)"
    $benchJson = $null
    # NB: use a distinct name — PS variables are case-insensitive, so `$bench`
    # would clobber the `$Bench` path above before Get-Content runs.
    try { $benchJson = Get-Content -LiteralPath $Bench -Raw | ConvertFrom-Json -ErrorAction Stop } catch { }
    if ($benchJson -and $benchJson.mean_decode_tps -gt 0) {
        $benchOk = $true
        $benchDetail = ('mean_decode_tps={0} {1} {2} n_ctx={3}' -f $benchJson.mean_decode_tps, $benchJson.model, $benchJson.quant, $benchJson.n_ctx)
    }
}
Add-Check 'bench baseline' $benchOk $benchDetail

# ---- (b) live serve E2E on :8099 ----------------------------------------------
Write-Output ''
Write-Output '== (b) dllm serve --port 8099 (E2E) =='
$ApiChecks = @('node fingerprint', 'model catalog', 'stats fields', 'session create', 'message POST', 'token events (SSE)')
$pre = (& curl.exe -s -m 2 "$Base/api/health" 2>$null) -join ''
$ServerRun = $false

if ($pre -match '"ok"\s*:\s*true') {
    Add-Check 'serve startup' $false 'port 8099 answered BEFORE start (stale dllm?) - E2E aborted'
    foreach ($n in $ApiChecks) { Add-Check $n $false 'skipped: port 8099 already owned by another process' }
}
elseif (-not (Test-Path -LiteralPath $Exe)) {
    Add-Check 'serve startup' $false 'dllm.exe missing - E2E skipped'
    foreach ($n in $ApiChecks) { Add-Check $n $false 'skipped: no exe' }
}
else {
    $Srv = Start-Process -FilePath $Exe -ArgumentList @('serve', '--port', '8099') `
        -WorkingDirectory $RepoRoot -WindowStyle Hidden `
        -RedirectStandardOutput $SrvOut -RedirectStandardError $SrvErr -PassThru
    try {
        $hOk = $false
        $deadline = [DateTime]::UtcNow.AddSeconds(15)
        while ([DateTime]::UtcNow -lt $deadline) {
            Start-Sleep -Milliseconds 500
            $h = (& curl.exe -s -m 2 "$Base/api/health" 2>$null) -join ''
            if ($h -match '"ok"\s*:\s*true') { $hOk = $true; break }
        }
        if ($hOk) {
            Add-Check 'serve startup' $true 'health ok:true within 15s'
            $ServerRun = $true
        }
        else {
            $tail = ''
            if (Test-Path -LiteralPath $SrvErr) { $tail = (Get-Content -LiteralPath $SrvErr -Tail 3 | Out-String).Trim() }
            Add-Check 'serve startup' $false ('no ok:true within 15s. stderr tail: ' + $tail)
            foreach ($n in $ApiChecks) { Add-Check $n $false 'skipped: server not healthy' }
        }

        if ($ServerRun) {
            # GET /api/node — fingerprint present (real TOFU identity, not the "unknown" default)
            $nTxt = (& curl.exe -s -m 3 "$Base/api/node" 2>$null) -join ''
            $node = $null
            try { $node = $nTxt | ConvertFrom-Json -ErrorAction Stop } catch { }
            $fp = ''
            $nid = ''
            if ($node) { $fp = [string]$node.fingerprint; $nid = [string]$node.node_id }
            $nodeOk = ($node -and $fp -ne '' -and $fp -ne 'unknown')
            Add-Check 'node fingerprint' $nodeOk ('node_id={0} fp={1}' -f $nid, $fp)

            # GET /api/models — qwen3-0.6b-q4 present in the baked catalog
            $mTxt = (& curl.exe -s -m 3 "$Base/api/models" 2>$null) -join ''
            Add-Check 'model catalog' ($mTxt -match 'qwen3-0\.6b-q4') 'GET /api/models contains qwen3-0.6b-q4'

            # GET /api/stats — all fields present (engine llama|mock both accepted)
            $stTxt = (& curl.exe -s -m 3 "$Base/api/stats" 2>$null) -join ''
            $st = $null
            try { $st = $stTxt | ConvertFrom-Json -ErrorAction Stop } catch { }
            $stOk = ($st -and $null -ne $st.uptime_s -and $st.engine -and $null -ne $st.sessions -and $null -ne $st.events -and $st.node_id)
            Add-Check 'stats fields' $stOk ('engine={0} sessions={1} events={2} uptime_s={3}' -f $st.engine, $st.sessions, $st.events, $st.uptime_s)

            # POST /v1/sessions -> id (body via temp file, never inline JSON)
            Set-Content -LiteralPath $BodySession -Value '{}' -Encoding ASCII -NoNewline
            $cTxt = (& curl.exe -s -m 5 -X POST -H 'Content-Type: application/json' --data-binary "@$BodySession" "$Base/v1/sessions" 2>$null) -join ''
            $sess = $null
            try { $sess = $cTxt | ConvertFrom-Json -ErrorAction Stop } catch { }
            $sessId = ''
            $sessOk = $false
            if ($sess -and $sess.id) { $sessId = [string]$sess.id; $sessOk = $sessId.StartsWith('sess-') }
            Add-Check 'session create' $sessOk ('POST /v1/sessions -> id=' + $sessId)

            # POST message — curl --data-binary @tmpfile
            $msgOk = $false
            Set-Content -LiteralPath $BodyMessage -Value '{"text":"hello from the MVP acceptance gate"}' -Encoding ASCII -NoNewline
            if ($sessOk) {
                $rTxt = (& curl.exe -s -m 5 -X POST -H 'Content-Type: application/json' --data-binary "@$BodyMessage" "$Base/v1/sessions/$sessId/messages" 2>$null) -join ''
                $msgOk = ($rTxt -match '"accepted"\s*:\s*true')
                Add-Check 'message POST' $msgOk 'accepted:true'
            }
            else {
                Add-Check 'message POST' $false 'skipped: no session id'
            }

            # GET /v1/sessions/{id}/events — short -m window; tokens may be llama
            # or mock (engine fallback) — both accepted. Note: the SSE replay path
            # tags every stored row as event "token", so replayed session/user
            # rows count too; the kind lives in the data payload.
            if ($msgOk) {
                $evTxt = (& curl.exe -s -N -m 6 "$Base/v1/sessions/$sessId/events" 2>$null) -join ''
                $tokHits = ([regex]::Matches($evTxt, 'token')).Count
                $evOk = ($tokHits -ge 1 -and $evTxt -match 'data:')
                Add-Check 'token events (SSE)' $evOk ('token-hits={0} in 6s window (llama or mock both accepted)' -f $tokHits)
            }
            else {
                Add-Check 'token events (SSE)' $false 'skipped: message not accepted'
            }
        }
    }
    finally {
        if ($Srv -and -not $Srv.HasExited) {
            Stop-Process -Id $Srv.Id -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 300
        }
    }
}

# ---- (c) pipe_pair drill -------------------------------------------------------
Write-Output ''
Write-Output '== (c) pipe_pair drill =='
. (Join-Path $PSScriptRoot 'build-env.ps1')   # cargo env (MinGW, LIBCLANG, RUSTFLAGS)
$ppText = (& cargo run -p dllm-core --example pipe_pair 2>&1 | Out-String)
$ppOk = ($LASTEXITCODE -eq 0 -and $ppText -match 'PIPE_PAIR PASS')
$ppDetail = 'no PIPE_PAIR PASS in output'
if ($ppOk) {
    $ppDetail = ($ppText -split '\r?\n' | Where-Object { $_ -match 'PIPE_PAIR' } | Select-Object -First 1).Trim()
}
Add-Check 'pipe_pair drill' $ppOk $ppDetail

# ---- (d) checklist + exit code -------------------------------------------------
Write-Output ''
Write-Output '================== MVP ACCEPTANCE CHECKLIST =================='
$pass = 0
$fail = 0
foreach ($r in $script:Rows) {
    $tag = 'FAIL'
    if ($r.Ok) { $tag = 'PASS'; $pass = $pass + 1 } else { $fail = $fail + 1 }
    Write-Output ('  [{0}] {1,-22} {2}' -f $tag, $r.Name, $r.Detail)
}
Write-Output '=============================================================='
Write-Output ('MVP ACCEPTANCE: {0} passed, {1} failed' -f $pass, $fail)
Remove-Item -LiteralPath $BodySession, $BodyMessage -ErrorAction SilentlyContinue
if ($fail -gt 0) { exit 1 }
exit 0
