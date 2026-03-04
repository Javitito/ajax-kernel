param(
  [string]$Root,
  [string]$Rail = "lab",
  [int]$Port = 5012,
  [switch]$EnsureWorker,
  [int]$TickIntervalSec = 30,
  [int]$MaxTicks = 1,
  [string]$StopFile
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
  param([int]$PortNumber)
  $publicRoot = if ($env:PUBLIC) { $env:PUBLIC } else { "C:\Users\Public" }
  $dir = Join-Path $publicRoot "ajax"
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  return Join-Path $dir ("watchdog_{0}_task.log" -f $PortNumber)
}

function Rotate-LogIfNeeded {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    return
  }
  try {
    $size = (Get-Item -LiteralPath $Path).Length
    if ($size -lt 1048576) {
      return
    }
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $rotated = $Path.Replace(".log", "_$stamp.log")
    Move-Item -LiteralPath $Path -Destination $rotated -Force
  } catch {
  }
}

function Write-PublicLog {
  param(
    [string]$LogPath,
    [string]$Message
  )
  $stamp = (Get-Date).ToUniversalTime().ToString("o")
  Add-Content -Path $LogPath -Encoding UTF8 -Value "[$stamp] $Message"
}

function Probe-Port {
  param(
    [int]$PortNumber,
    [string]$RailName
  )
  $host = "127.0.0.1"
  $result = [ordered]@{
    port = $PortNumber
    listener = $false
    health_ok = $false
    displays_ok = if ($RailName -eq "lab") { $false } else { $null }
    health_error = $null
    displays_error = $null
  }
  try {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $PortNumber -ErrorAction SilentlyContinue
    if ($listener) {
      $result.listener = $true
    }
  } catch {
    $result.listener = $false
  }
  if (-not $result.listener) {
    $result.health_error = "listener_down"
    if ($RailName -eq "lab") {
      $result.displays_error = "listener_down"
    }
    return $result
  }
  try {
    $health = Invoke-RestMethod -Method Get -Uri ("http://{0}:{1}/health" -f $host, $PortNumber) -TimeoutSec 2 -ErrorAction Stop
    $result.health_ok = [bool]$health.ok
    if (-not $result.health_ok) {
      $result.health_error = "health_not_ok"
    }
  } catch {
    $result.health_error = $_.Exception.Message
    if ($_.Exception.Message -match "401|403") {
      $result.health_ok = $true
      $result.health_error = "health_auth_required"
    }
  }
  if ($RailName -eq "lab") {
    try {
      $displays = Invoke-RestMethod -Method Get -Uri ("http://{0}:{1}/displays" -f $host, $PortNumber) -TimeoutSec 2 -ErrorAction Stop
      $count = 0
      if ($displays -and $displays.displays) {
        $count = @($displays.displays).Count
      }
      $result.displays_ok = $count -ge 1
      if (-not $result.displays_ok) {
        $result.displays_error = "display_catalog_empty"
      }
    } catch {
      $result.displays_error = $_.Exception.Message
      if ($_.Exception.Message -match "401|403") {
        $result.displays_ok = $true
        $result.displays_error = "displays_auth_required"
      }
    }
  }
  return $result
}

function Decide-WatchdogAction {
  param(
    [object]$Probe,
    [string]$RailName
  )
  $reasons = @()
  if (-not [bool]$Probe.listener) {
    $reasons += "listener_down"
  }
  if (-not [bool]$Probe.health_ok) {
    $reasons += "health_not_ok"
  }
  if ($RailName -eq "lab" -and -not [bool]$Probe.displays_ok) {
    $reasons += "display_catalog_unavailable"
  }
  return [ordered]@{
    would_start = ($reasons.Count -gt 0)
    reasons = $reasons
  }
}

function Invoke-LabBootstrap {
  param(
    [string]$RepoRoot,
    [switch]$EnsureLabWorker
  )
  $script = Join-Path $PSScriptRoot "ajax_lab_bootstrap.ps1"
  if (-not (Test-Path -LiteralPath $script)) {
    return [ordered]@{
      ok = $false
      error = "missing_ajax_lab_bootstrap.ps1"
      returncode = 2
    }
  }
  $cmd = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $script,
    "-Root",
    $RepoRoot,
    "-Rail",
    "lab"
  )
  if ($EnsureLabWorker) {
    $cmd += "-EnsureWorker"
  }
  try {
    $raw = & (Join-Path $PSHOME "powershell.exe") @cmd 2>&1 | Out-String
    $payload = $null
    try { $payload = $raw | ConvertFrom-Json } catch { $payload = $null }
    return [ordered]@{
      ok = if ($payload) { [bool]$payload.ok } else { $false }
      returncode = if ($payload -and $payload.ok) { 0 } else { 2 }
      payload = $payload
      raw = $raw.Trim()
    }
  } catch {
    return [ordered]@{
      ok = $false
      error = $_.Exception.Message
      returncode = 2
    }
  }
}

function Invoke-ProdBootstrap {
  param([string]$RepoRoot)
  $startScript = Join-Path $RepoRoot "Start-AjaxDriver.ps1"
  if (-not (Test-Path -LiteralPath $startScript)) {
    return [ordered]@{
      ok = $false
      returncode = 2
      error = "missing_Start-AjaxDriver.ps1"
    }
  }
  try {
    & (Join-Path $PSHOME "powershell.exe") -NoProfile -ExecutionPolicy Bypass -File $startScript -Port 5010 | Out-Null
    return [ordered]@{
      ok = $true
      returncode = 0
    }
  } catch {
    return [ordered]@{
      ok = $false
      returncode = 2
      error = $_.Exception.Message
    }
  }
}

function Invoke-AnchorCheck {
  param(
    [string]$RepoRoot,
    [string]$RailName
  )
  try {
    Push-Location $RepoRoot
    $raw = (& python "bin/ajaxctl" "doctor" "anchor" "--rail" $RailName 2>&1 | Out-String)
    Pop-Location
    try {
      return ($raw | ConvertFrom-Json)
    } catch {
      return [ordered]@{
        ok = $false
        error = "anchor_output_not_json"
        raw = $raw.Trim()
      }
    }
  } catch {
    try { Pop-Location } catch {}
    return [ordered]@{
      ok = $false
      error = $_.Exception.Message
    }
  }
}

$railNorm = if ($Rail) { $Rail.Trim().ToLowerInvariant() } else { "lab" }
if ($railNorm -ne "lab" -and $railNorm -ne "prod") {
  $railNorm = "lab"
}
$repoRoot = Resolve-RepoRoot -ExplicitRoot $Root
if (-not $StopFile -or -not $StopFile.Trim()) {
  $publicRoot = if ($env:PUBLIC) { $env:PUBLIC } else { "C:\Users\Public" }
  $StopFile = Join-Path (Join-Path $publicRoot "ajax") "watchdog.stop"
}
$logPath = Resolve-PublicLogPath -PortNumber $Port
Rotate-LogIfNeeded -Path $logPath

$tick = 0
$lastPayload = $null
$keepRunning = $true

while ($keepRunning) {
  $tick += 1
  if (Test-Path -LiteralPath $StopFile) {
    Write-PublicLog -LogPath $logPath -Message "stopfile_detected path=$StopFile"
    break
  }

  $before = Probe-Port -PortNumber $Port -RailName $railNorm
  $decision = Decide-WatchdogAction -Probe $before -RailName $railNorm
  $bootstrap = [ordered]@{
    attempted = $false
    ok = $true
    mode = "noop"
    detail = $null
  }
  if ([bool]$decision.would_start) {
    $bootstrap.attempted = $true
    if ($railNorm -eq "lab") {
      $bootstrap.mode = "lab_bootstrap"
      $labResult = Invoke-LabBootstrap -RepoRoot $repoRoot -EnsureLabWorker:$EnsureWorker
      $bootstrap.ok = [bool]$labResult.ok
      $bootstrap.detail = $labResult
    } else {
      $bootstrap.mode = "prod_start_driver"
      $prodResult = Invoke-ProdBootstrap -RepoRoot $repoRoot
      $bootstrap.ok = [bool]$prodResult.ok
      $bootstrap.detail = $prodResult
    }
  }

  Start-Sleep -Seconds 2
  $after = Probe-Port -PortNumber $Port -RailName $railNorm
  $anchor = if ($railNorm -eq "lab") { Invoke-AnchorCheck -RepoRoot $repoRoot -RailName "lab" } else { $null }

  $verified = [bool]$after.listener -and [bool]$after.health_ok
  if ($railNorm -eq "lab") {
    $verified = $verified -and [bool]$after.displays_ok
  }
  $status = "READY"
  if (-not $verified) {
    $status = "FAIL_CLOSED"
  }
  if ($railNorm -eq "lab" -and $anchor -and -not [bool]$anchor.ok) {
    $status = "FAIL_CLOSED"
    $verified = $false
  }

  $ts = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
  $opsDir = Join-Path $repoRoot ("artifacts\ops\{0}" -f $ts)
  New-Item -ItemType Directory -Force -Path $opsDir | Out-Null
  $receiptPath = Join-Path $opsDir "watchdog_tick.json"
  $lastPayload = [ordered]@{
    schema = "ajax.watchdog.tick.v1"
    ts_utc = (Get-Date).ToUniversalTime().ToString("o")
    tick = $tick
    rail = $railNorm
    port = $Port
    ok = $verified
    status = $status
    decision = $decision
    before = $before
    bootstrap = $bootstrap
    after = $after
    anchor = $anchor
    receipt_path = $receiptPath
    public_log = $logPath
    next_hint = @(
      "python bin/ajaxctl doctor boot --rail $railNorm",
      "python bin/ajaxctl ops tasks audit",
      "python bin/ajaxctl ops tasks ensure --apply"
    )
  }
  $lastPayload | ConvertTo-Json -Depth 18 | Set-Content -Path $receiptPath -Encoding UTF8
  Write-PublicLog -LogPath $logPath -Message "tick=$tick rail=$railNorm port=$Port status=$status decision=$($decision.reasons -join ',') receipt=$receiptPath"

  if ($MaxTicks -gt 0 -and $tick -ge $MaxTicks) {
    $keepRunning = $false
  } else {
    Start-Sleep -Seconds ([Math]::Max(1, $TickIntervalSec))
  }
}

if ($null -eq $lastPayload) {
  $lastPayload = [ordered]@{
    schema = "ajax.watchdog.tick.v1"
    ts_utc = (Get-Date).ToUniversalTime().ToString("o")
    tick = $tick
    rail = $railNorm
    port = $Port
    ok = $true
    status = "STOPPED"
    decision = [ordered]@{ would_start = $false; reasons = @("stopfile_detected") }
    next_hint = @("Remove stopfile and rerun watchdog task.")
  }
}

$lastPayload | ConvertTo-Json -Depth 18
if ([bool]$lastPayload.ok) {
  exit 0
}
exit 2
