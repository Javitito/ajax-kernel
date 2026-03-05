param(
  [string]$Root,
  [string]$Rail = "lab",
  [switch]$EnsureWorker,
  [switch]$SkipAnchorCheck,
  [int]$WorkerStaleSec = 30
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
  param([string]$ExplicitRoot)
  if ($ExplicitRoot -and $ExplicitRoot.Trim()) {
    return [System.IO.Path]::GetFullPath($ExplicitRoot)
  }
  $candidate = Join-Path $PSScriptRoot "..\.."
  return [System.IO.Path]::GetFullPath($candidate)
}

function Resolve-PublicLogPath {
  $publicRoot = if ($env:PUBLIC) { $env:PUBLIC } else { "C:\Users\Public" }
  $dir = Join-Path $publicRoot "ajax"
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  return Join-Path $dir "lab_bootstrap_task.log"
}

function Write-PublicLog {
  param(
    [string]$LogPath,
    [string]$Message
  )
  $stamp = (Get-Date).ToUniversalTime().ToString("o")
  Add-Content -Path $LogPath -Encoding UTF8 -Value "[$stamp] $Message"
}

function Probe-Port5012 {
  $out = [ordered]@{
    listener = $false
    health_ok = $false
    displays_ok = $false
    health_error = $null
    displays_error = $null
  }
  try {
    $listener = Get-NetTCPConnection -State Listen -LocalPort 5012 -ErrorAction SilentlyContinue
    if ($listener) {
      $out.listener = $true
    }
  } catch {
    $out.listener = $false
  }
  if (-not $out.listener) {
    $out.health_error = "listener_down"
    $out.displays_error = "listener_down"
    return $out
  }
  try {
    $health = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:5012/health" -TimeoutSec 2 -ErrorAction Stop
    $out.health_ok = [bool]$health.ok
    if (-not $out.health_ok) {
      $out.health_error = "health_not_ok"
    }
  } catch {
    $out.health_error = $_.Exception.Message
    if ($_.Exception.Message -match "401|403") {
      $out.health_ok = $true
      $out.health_error = "health_auth_required"
    }
  }
  try {
    $displays = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:5012/displays" -TimeoutSec 2 -ErrorAction Stop
    $count = 0
    if ($displays -and $displays.displays) {
      $count = @($displays.displays).Count
    }
    $out.displays_ok = $count -ge 1
    if (-not $out.displays_ok) {
      $out.displays_error = "display_catalog_empty"
    }
  } catch {
    $out.displays_error = $_.Exception.Message
    if ($_.Exception.Message -match "401|403") {
      $out.displays_ok = $true
      $out.displays_error = "displays_auth_required"
    }
  }
  return $out
}

function Get-WorkerState {
  param(
    [string]$RepoRoot,
    [int]$StaleSec
  )
  $labDir = Join-Path $RepoRoot "artifacts\lab"
  $heartbeatPath = Join-Path $labDir "heartbeat.json"
  $pidPath = Join-Path $labDir "worker.pid"
  $workerPid = $null
  $running = $false
  $heartbeatAge = $null
  if (Test-Path -LiteralPath $pidPath) {
    try {
      $workerPid = [int](Get-Content -Raw -Encoding UTF8 $pidPath).Trim()
      $proc = Get-Process -Id $workerPid -ErrorAction SilentlyContinue
      if ($proc) {
        $running = $true
      }
    } catch {
      $workerPid = $null
      $running = $false
    }
  }
  if (Test-Path -LiteralPath $heartbeatPath) {
    try {
      $hb = Get-Content -Raw -Encoding UTF8 $heartbeatPath | ConvertFrom-Json
      if ($hb.ts) {
        $hbTs = [double]$hb.ts
        $nowTs = [double][DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        $heartbeatAge = [math]::Max(0, ($nowTs - $hbTs))
      }
    } catch {
      $heartbeatAge = $null
    }
  }
  $fresh = $false
  if ($running -and $null -ne $heartbeatAge -and $heartbeatAge -le $StaleSec) {
    $fresh = $true
  }
  return [ordered]@{
    fresh = $fresh
    running = $running
    pid = $workerPid
    heartbeat_age_s = $heartbeatAge
    stale_threshold_s = $StaleSec
    heartbeat_path = $heartbeatPath
    pid_path = $pidPath
    status = if ($fresh) { "fresh" } elseif ($running) { "stale" } else { "down" }
  }
}

$repoRoot = Resolve-RepoRoot -ExplicitRoot $Root
$logPath = Resolve-PublicLogPath
$ts = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$artifactDir = Join-Path $repoRoot "artifacts\boot"
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$artifactPath = Join-Path $artifactDir "lab_boot_$ts.json"

$beforeProbe = Probe-Port5012
$driverStart = [ordered]@{
  attempted = $false
  returncode = 0
  error = $null
}

if (-not $beforeProbe.listener -or -not $beforeProbe.health_ok) {
  $driverStart.attempted = $true
  $startScript = Join-Path $repoRoot "Start-AjaxDriver.ps1"
  if (-not (Test-Path -LiteralPath $startScript)) {
    $driverStart.returncode = 2
    $driverStart.error = "missing_Start-AjaxDriver.ps1"
  } else {
    try {
      & (Join-Path $PSHOME "powershell.exe") -NoProfile -ExecutionPolicy Bypass -File $startScript -Port 5012 | Out-Null
      $driverStart.returncode = 0
    } catch {
      $driverStart.returncode = 2
      $driverStart.error = $_.Exception.Message
    }
  }
}

Start-Sleep -Seconds 2

$workerBefore = Get-WorkerState -RepoRoot $repoRoot -StaleSec $WorkerStaleSec
$workerStart = [ordered]@{
  attempted = $false
  returncode = 0
  error = $null
}

if ($EnsureWorker -and -not $workerBefore.fresh) {
  $workerStart.attempted = $true
  try {
    Push-Location $repoRoot
    & python "bin/ajaxctl" "lab" "start" 2>&1 | Out-Null
    $workerStart.returncode = 0
  } catch {
    $workerStart.returncode = 2
    $workerStart.error = $_.Exception.Message
  } finally {
    Pop-Location
  }
}

Start-Sleep -Seconds 2
$afterProbe = Probe-Port5012
$workerAfter = Get-WorkerState -RepoRoot $repoRoot -StaleSec $WorkerStaleSec

$anchor = $null
if (-not $SkipAnchorCheck) {
  try {
    Push-Location $repoRoot
    $anchorRaw = (& python "bin/ajaxctl" "doctor" "anchor" "--rail" "lab" 2>&1 | Out-String)
    Pop-Location
    try {
      $anchor = $anchorRaw | ConvertFrom-Json
    } catch {
      $anchor = [ordered]@{
        ok = $false
        error = "anchor_output_not_json"
        raw = $anchorRaw.Trim()
      }
    }
  } catch {
    try { Pop-Location } catch {}
    $anchor = [ordered]@{
      ok = $false
      error = $_.Exception.Message
    }
  }
}

$ok = [bool]$afterProbe.listener -and [bool]$afterProbe.health_ok -and [bool]$afterProbe.displays_ok
if ($EnsureWorker) {
  $ok = $ok -and [bool]$workerAfter.fresh
}
if ($null -ne $anchor) {
  $ok = $ok -and [bool]$anchor.ok
}

$status = if ($ok) { "READY" } else { "FAIL_CLOSED" }
$nextHint = @(
  "python bin/ajaxctl doctor anchor --rail lab",
  "python bin/ajaxctl doctor boot --rail lab",
  "python bin/ajaxctl ops tasks ensure --apply"
)

$payload = [ordered]@{
  schema = "ajax.boot.lab_bootstrap.v1"
  ts_utc = (Get-Date).ToUniversalTime().ToString("o")
  rail = $Rail
  ok = $ok
  status = $status
  repo_root = $repoRoot
  before = $beforeProbe
  after = $afterProbe
  worker_before = $workerBefore
  worker_after = $workerAfter
  driver_start = $driverStart
  worker_start = $workerStart
  anchor = $anchor
  next_hint = $nextHint
  artifact_path = $artifactPath
  public_log = $logPath
}

$payload | ConvertTo-Json -Depth 16 | Set-Content -Path $artifactPath -Encoding UTF8
Write-PublicLog -LogPath $logPath -Message "lab_bootstrap status=$status ok=$ok"
$payload | ConvertTo-Json -Depth 16

if ($ok) {
  exit 0
}
exit 2
