param(
  [Alias("Host")]
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 5010,
  [string]$Root,
  [string]$PythonExe,
  [string]$DriverScript,
  [int]$WaitSeconds = 9,
  [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
  param([string]$ExplicitRoot)
  if ($ExplicitRoot -and $ExplicitRoot.Trim()) {
    return [System.IO.Path]::GetFullPath($ExplicitRoot)
  }
  return [System.IO.Path]::GetFullPath($PSScriptRoot)
}

function Resolve-PythonLauncher {
  param(
    [string]$RepoRoot,
    [string]$ExplicitPython
  )
  if ($ExplicitPython -and $ExplicitPython.Trim()) {
    if (Test-Path -LiteralPath $ExplicitPython) {
      return [ordered]@{
        exists = $true
        path = [System.IO.Path]::GetFullPath($ExplicitPython)
        source = "arg:PythonExe"
      }
    }
    return [ordered]@{
      exists = $false
      path = $ExplicitPython
      source = "arg:PythonExe"
    }
  }

  $venvPython = Join-Path $RepoRoot ".venv_os_driver\Scripts\python.exe"
  if (Test-Path -LiteralPath $venvPython) {
    return [ordered]@{
      exists = $true
      path = [System.IO.Path]::GetFullPath($venvPython)
      source = "repo:.venv_os_driver"
    }
  }

  foreach ($candidate in @("python.exe", "python", "py.exe", "py")) {
    try {
      $cmd = Get-Command $candidate -ErrorAction Stop | Select-Object -First 1
      if ($cmd -and $cmd.Source) {
        return [ordered]@{
          exists = $true
          path = $cmd.Source
          source = "path:$candidate"
        }
      }
    } catch {
    }
  }

  return [ordered]@{
    exists = $false
    path = $null
    source = "missing"
  }
}

function Resolve-LaunchTarget {
  param(
    [string]$RepoRoot,
    [string]$TargetHost,
    [int]$PortNumber,
    [string]$ExplicitDriverScript
  )
  if ($PortNumber -eq 5012) {
    $targetPath = Join-Path $RepoRoot "agency\lab_dummy_driver.py"
    return [ordered]@{
      rail = "lab"
      target_kind = "python_module"
      target_label = "agency.lab_dummy_driver"
      resolved_target = $targetPath
      exists = Test-Path -LiteralPath $targetPath
      arguments = @(
        "-m",
        "agency.lab_dummy_driver",
        "--serve",
        "--root",
        $RepoRoot,
        "--host",
        $TargetHost,
        "--port",
        [string]$PortNumber
      )
      missing_reason = "missing_lab_entrypoint"
    }
  }

  $candidate = if ($ExplicitDriverScript -and $ExplicitDriverScript.Trim()) {
    $ExplicitDriverScript
  } else {
    Join-Path $RepoRoot "drivers\os_driver.py"
  }
  return [ordered]@{
    rail = "prod"
    target_kind = "python_script"
    target_label = "drivers.os_driver"
    resolved_target = $candidate
    exists = Test-Path -LiteralPath $candidate
    arguments = @(
      $candidate,
      "--host",
      $TargetHost,
      "--port",
      [string]$PortNumber
    )
    missing_reason = "missing_driver_entrypoint"
  }
}

function Test-DriverHealth {
  param(
    [string]$ProbeHost,
    [int]$ProbePort
  )
  $url = "http://{0}:{1}/health" -f $ProbeHost, $ProbePort
  try {
    $resp = Invoke-WebRequest -Method Get -Uri $url -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
    $payload = $null
    if ($resp.Content) {
      try {
        $payload = $resp.Content | ConvertFrom-Json -ErrorAction Stop
      } catch {
        $payload = $null
      }
    }
    $healthy = $true
    $detail = "health_ok"
    if ($payload -and $null -ne $payload.ok) {
      $healthy = [bool]$payload.ok
      if (-not $healthy) {
        $detail = "health_not_ok"
      }
    }
    return [ordered]@{
      healthy = $healthy
      detail = $detail
      http_status = [int]$resp.StatusCode
      url = $url
    }
  } catch {
    $statusCode = $null
    try {
      $statusCode = [int]$_.Exception.Response.StatusCode.value__
    } catch {
      $statusCode = $null
    }
    if ($statusCode -eq 401 -or $statusCode -eq 403) {
      return [ordered]@{
        healthy = $true
        detail = "auth_challenge"
        http_status = $statusCode
        url = $url
      }
    }
    return [ordered]@{
      healthy = $false
      detail = $_.Exception.Message
      http_status = $statusCode
      url = $url
    }
  }
}

function Read-LogTail {
  param([string]$Path)
  if (-not $Path -or -not (Test-Path -LiteralPath $Path)) {
    return $null
  }
  try {
    $lines = Get-Content -LiteralPath $Path -Encoding UTF8 -Tail 8 -ErrorAction Stop
    return ($lines -join "`n")
  } catch {
    return $null
  }
}

function Stop-ChildProcess {
  param($Process)
  if ($null -eq $Process) {
    return $false
  }
  try {
    if (-not $Process.HasExited) {
      Stop-Process -Id $Process.Id -Force -ErrorAction Stop
      return $true
    }
  } catch {
  }
  return $false
}

function Write-LauncherResult {
  param([hashtable]$Payload)
  $Payload | ConvertTo-Json -Depth 10 -Compress
}

$repoRoot = Resolve-RepoRoot -ExplicitRoot $Root
$artifactDir = Join-Path $repoRoot "artifacts\driver\launcher"
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$ts = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$stdoutLog = Join-Path $artifactDir ("prod_driver_{0}_{1}.stdout.log" -f $Port, $ts)
$stderrLog = Join-Path $artifactDir ("prod_driver_{0}_{1}.stderr.log" -f $Port, $ts)

$python = Resolve-PythonLauncher -RepoRoot $repoRoot -ExplicitPython $PythonExe
$target = Resolve-LaunchTarget -RepoRoot $repoRoot -TargetHost $BindHost -PortNumber $Port -ExplicitDriverScript $DriverScript
$preHealth = Test-DriverHealth -ProbeHost $BindHost -ProbePort $Port

if (-not [bool]$ForceRestart -and [bool]$preHealth.healthy) {
  Write-LauncherResult @{
    schema = "ajax.driver_launcher.v1"
    ok = $true
    status = "already_healthy"
    rail = $target.rail
    host = $BindHost
    port = $Port
    python = $python.path
    python_source = $python.source
    resolved_target = $target.resolved_target
    target_kind = $target.target_kind
    launch_attempted = $false
    pre_health = $preHealth
    post_health = $preHealth
  }
  exit 0
}

if (-not [bool]$python.exists) {
  Write-LauncherResult @{
    schema = "ajax.driver_launcher.v1"
    ok = $false
    status = "prereq_failed"
    reason = "missing_python_launcher"
    rail = $target.rail
    host = $BindHost
    port = $Port
    python = $python.path
    python_source = $python.source
    resolved_target = $target.resolved_target
    target_kind = $target.target_kind
    launch_attempted = $false
    pre_health = $preHealth
    post_health = $null
  }
  exit 2
}

if (-not [bool]$target.exists) {
  Write-LauncherResult @{
    schema = "ajax.driver_launcher.v1"
    ok = $false
    status = "prereq_failed"
    reason = $target.missing_reason
    rail = $target.rail
    host = $BindHost
    port = $Port
    python = $python.path
    python_source = $python.source
    resolved_target = $target.resolved_target
    target_kind = $target.target_kind
    launch_attempted = $false
    pre_health = $preHealth
    post_health = $null
  }
  exit 3
}

$process = $null
try {
  $process = Start-Process `
    -FilePath $python.path `
    -ArgumentList $target.arguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru
} catch {
  Write-LauncherResult @{
    schema = "ajax.driver_launcher.v1"
    ok = $false
    status = "launch_failed"
    reason = "start_process_failed"
    detail = $_.Exception.Message
    rail = $target.rail
    host = $BindHost
    port = $Port
    python = $python.path
    python_source = $python.source
    resolved_target = $target.resolved_target
    target_kind = $target.target_kind
    launch_attempted = $true
    pre_health = $preHealth
    post_health = $null
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
  }
  exit 4
}

$deadline = (Get-Date).ToUniversalTime().AddSeconds([Math]::Max(1, $WaitSeconds))
$postHealth = $preHealth
while ((Get-Date).ToUniversalTime() -lt $deadline) {
  Start-Sleep -Milliseconds 750
  $postHealth = Test-DriverHealth -ProbeHost $BindHost -ProbePort $Port
  if ([bool]$postHealth.healthy) {
    Write-LauncherResult @{
      schema = "ajax.driver_launcher.v1"
      ok = $true
      status = "healthy"
      rail = $target.rail
      host = $BindHost
      port = $Port
      pid = $process.Id
      python = $python.path
      python_source = $python.source
      resolved_target = $target.resolved_target
      target_kind = $target.target_kind
      launch_attempted = $true
      pre_health = $preHealth
      post_health = $postHealth
      stdout_log = $stdoutLog
      stderr_log = $stderrLog
    }
    exit 0
  }
  if ($process.HasExited) {
    Write-LauncherResult @{
      schema = "ajax.driver_launcher.v1"
      ok = $false
      status = "launch_failed"
      reason = "driver_process_exited"
      rail = $target.rail
      host = $BindHost
      port = $Port
      pid = $process.Id
      returncode = $process.ExitCode
      python = $python.path
      python_source = $python.source
      resolved_target = $target.resolved_target
      target_kind = $target.target_kind
      launch_attempted = $true
      pre_health = $preHealth
      post_health = $postHealth
      stdout_log = $stdoutLog
      stderr_log = $stderrLog
      stdout_tail = Read-LogTail -Path $stdoutLog
      stderr_tail = Read-LogTail -Path $stderrLog
    }
    exit 5
  }
}

$stopped = Stop-ChildProcess -Process $process
$postHealth = Test-DriverHealth -ProbeHost $BindHost -ProbePort $Port
Write-LauncherResult @{
  schema = "ajax.driver_launcher.v1"
  ok = $false
  status = "launch_timeout"
  reason = "health_timeout"
  rail = $target.rail
  host = $BindHost
  port = $Port
  pid = $process.Id
  python = $python.path
  python_source = $python.source
  resolved_target = $target.resolved_target
  target_kind = $target.target_kind
  launch_attempted = $true
  pre_health = $preHealth
  post_health = $postHealth
  timeout_s = [Math]::Max(1, $WaitSeconds)
  killed_process = $stopped
  stdout_log = $stdoutLog
  stderr_log = $stderrLog
  stdout_tail = Read-LogTail -Path $stdoutLog
  stderr_tail = Read-LogTail -Path $stderrLog
}
exit 6
