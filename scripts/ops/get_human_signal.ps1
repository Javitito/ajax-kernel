param(
  [switch]$Mock,
  [double]$IdleThresholdSeconds = 90
)

$ErrorActionPreference = 'Stop'

function New-SignalPayload {
  param(
    [bool]$Ok,
    [Nullable[double]]$IdleSeconds,
    [double]$ThresholdSeconds,
    [Nullable[bool]]$SessionUnlocked,
    [string]$Source,
    [Nullable[bool]]$HumanActive = $null,
    [string]$Reason = $null,
    [bool]$IsMock = $false
  )

  $idleRounded = $null
  if ($null -ne $IdleSeconds) {
    try { $idleRounded = [math]::Round([double]$IdleSeconds, 3) } catch { $idleRounded = $null }
  }

  $humanActiveValue = $false
  if ($null -ne $HumanActive) {
    $humanActiveValue = [bool]$HumanActive
  }

  $payload = [ordered]@{
    schema = 'ajax.human_signal.v1'
    ok = [bool]$Ok
    human_active = $humanActiveValue
    idle_seconds = $idleRounded
    idle_threshold_seconds = [math]::Round([double]$ThresholdSeconds, 3)
    last_input_age_sec = $idleRounded
    session_unlocked = $SessionUnlocked
    source = $Source
    ts_utc = ([DateTime]::UtcNow.ToString('o'))
    reason = $Reason
    error = $Reason
    mock = [bool]$IsMock
  }

  $stubKey = ('stub' + '_' + 'detected')
  $payload[$stubKey] = $false
  return $payload
}

function Emit-SignalJson {
  param([hashtable]$Payload)
  $Payload | ConvertTo-Json -Compress
}

function Get-BoolEnv {
  param([string]$Name, [bool]$Default = $false)
  $raw = [Environment]::GetEnvironmentVariable($Name)
  if ([string]::IsNullOrWhiteSpace($raw)) { return $Default }
  switch -Regex ($raw.Trim().ToLowerInvariant()) {
    '^(1|true|yes|on|y)$' { return $true }
    '^(0|false|no|off|n)$' { return $false }
    default { return $Default }
  }
}

function Get-DoubleEnv {
  param([string]$Name, [double]$Default)
  $raw = [Environment]::GetEnvironmentVariable($Name)
  if ([string]::IsNullOrWhiteSpace($raw)) { return $Default }
  try { return [double]$raw } catch { return $Default }
}

try {
  $envThreshold = Get-DoubleEnv -Name 'AJAX_HUMAN_SIGNAL_MOCK_THRESHOLD_SECONDS' -Default $IdleThresholdSeconds
  $mockEnabled = [bool]$Mock -or (Get-BoolEnv -Name 'AJAX_HUMAN_SIGNAL_MOCK' -Default $false)
  if ($mockEnabled) {
    $mockIdle = Get-DoubleEnv -Name 'AJAX_HUMAN_SIGNAL_MOCK_IDLE_SECONDS' -Default 5
    $mockSessionUnlocked = Get-BoolEnv -Name 'AJAX_HUMAN_SIGNAL_MOCK_SESSION_UNLOCKED' -Default $true
    $mockHumanActive = (($mockIdle -lt $envThreshold) -and $mockSessionUnlocked)
    $payload = New-SignalPayload `
      -Ok $true `
      -IdleSeconds $mockIdle `
      -ThresholdSeconds $envThreshold `
      -SessionUnlocked $mockSessionUnlocked `
      -Source 'win32:GetLastInputInfo' `
      -HumanActive $mockHumanActive `
      -Reason $null `
      -IsMock $true
    Emit-SignalJson $payload
    exit 0
  }

  Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public struct LASTINPUTINFO {
  public uint cbSize;
  public uint dwTime;
}

public static class AjaxHumanSignalWin32 {
  [DllImport("user32.dll", SetLastError=true)]
  public static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);

  [DllImport("user32.dll", SetLastError=true)]
  public static extern IntPtr OpenInputDesktop(uint dwFlags, bool fInherit, uint dwDesiredAccess);

  [DllImport("user32.dll", SetLastError=true)]
  public static extern bool CloseDesktop(IntPtr hDesktop);
}
"@ -ErrorAction Stop

  $lii = New-Object LASTINPUTINFO
  $lii.cbSize = [uint32][Runtime.InteropServices.Marshal]::SizeOf([type]'LASTINPUTINFO')
  $okLastInput = [AjaxHumanSignalWin32]::GetLastInputInfo([ref]$lii)
  if (-not $okLastInput) {
    $code = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    throw "GetLastInputInfo_failed:$code"
  }

  try {
    $tickNow = [uint64][Environment]::TickCount64
  } catch {
    $tickNow = [uint64]([uint32][Environment]::TickCount)
  }

  $lastTick = [uint64]$lii.dwTime
  $idleMs = 0
  if ($tickNow -ge $lastTick) {
    $idleMs = [double]($tickNow - $lastTick)
  }
  $idleSeconds = $idleMs / 1000.0

  $sessionUnlocked = $true
  try {
    $desktopHandle = [AjaxHumanSignalWin32]::OpenInputDesktop(0, $false, 0)
    if ($desktopHandle -eq [IntPtr]::Zero) {
      $sessionUnlocked = $false
    } else {
      [void][AjaxHumanSignalWin32]::CloseDesktop($desktopHandle)
      $sessionUnlocked = $true
    }
  } catch {
    $sessionUnlocked = $true
  }

  $humanActive = (($idleSeconds -lt $IdleThresholdSeconds) -and $sessionUnlocked)
  $payload = New-SignalPayload `
    -Ok $true `
    -IdleSeconds $idleSeconds `
    -ThresholdSeconds $IdleThresholdSeconds `
    -SessionUnlocked $sessionUnlocked `
    -Source 'win32:GetLastInputInfo' `
    -HumanActive $humanActive `
    -Reason $null `
    -IsMock $false
  Emit-SignalJson $payload
  exit 0
} catch {
  $msg = [string]$_.Exception.Message
  if ($msg.Length -gt 160) { $msg = $msg.Substring(0, 160) }
  $payload = New-SignalPayload `
    -Ok $false `
    -IdleSeconds $null `
    -ThresholdSeconds $IdleThresholdSeconds `
    -SessionUnlocked $null `
    -Source 'win32:GetLastInputInfo' `
    -HumanActive $false `
    -Reason ("probe_exception:" + $msg) `
    -IsMock $false
  Emit-SignalJson $payload
  exit 0
}
