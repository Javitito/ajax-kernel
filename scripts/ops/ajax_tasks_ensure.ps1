param(
  [string]$Root,
  [string]$ManifestPath,
  [switch]$DryRun,
  [switch]$Apply
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

function Normalize-PathText {
  param([string]$Value)
  if (-not $Value) {
    return ""
  }
  return $Value.ToLowerInvariant().Replace("/", "\")
}

function Split-TaskName {
  param([string]$TaskFullName)
  $raw = [string]$TaskFullName
  if (-not $raw.StartsWith("\")) {
    throw "task_name_invalid:$raw"
  }
  $parts = $raw.Split("\") | Where-Object { $_ -ne "" }
  if ($parts.Count -lt 2) {
    throw "task_name_invalid:$raw"
  }
  $leaf = $parts[-1]
  $path = "\" + (($parts[0..($parts.Count - 2)] -join "\")) + "\"
  return @{ TaskName = $leaf; TaskPath = $path }
}

function Parse-IsoDurationToTimeSpan {
  param([string]$IsoDuration)
  if (-not $IsoDuration) {
    return [TimeSpan]::FromMinutes(1)
  }
  $raw = $IsoDuration.Trim().ToUpperInvariant()
  if ($raw -notmatch "^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$") {
    return [TimeSpan]::FromMinutes(1)
  }
  $h = if ($matches[1]) { [int]$matches[1] } else { 0 }
  $m = if ($matches[2]) { [int]$matches[2] } else { 0 }
  $s = if ($matches[3]) { [int]$matches[3] } else { 0 }
  return New-TimeSpan -Hours $h -Minutes $m -Seconds $s
}

function Evaluate-TaskDrift {
  param(
    [object]$Task,
    [object]$TaskDef,
    [string]$ExpectedScriptNorm
  )
  $drift = @()
  if ($null -eq $Task) {
    $drift += "missing_task"
    return $drift
  }
  $enabled = $true
  try { $enabled = [bool]$Task.Settings.Enabled } catch { $enabled = $true }
  if (-not $enabled) {
    $drift += "disabled"
  }

  $runLevel = [string]$Task.Principal.RunLevel
  if ([string]$TaskDef.settings.run_with_highest -eq "True" -and $runLevel -notmatch "Highest") {
    $drift += "run_level_not_highest"
  }

  $expectedUser = [string]$TaskDef.trigger.user
  $runAs = [string]$Task.Principal.UserId
  if ($expectedUser -and $runAs -and $runAs.ToLowerInvariant() -notlike "*$($expectedUser.ToLowerInvariant())*") {
    $drift += "run_as_user_mismatch"
  }

  $actionBlob = ""
  foreach ($act in @($Task.Actions)) {
    $actionBlob = ([string]$act.Execute + " " + [string]$act.Arguments).Trim()
    break
  }
  if ($ExpectedScriptNorm -and (Normalize-PathText $actionBlob) -notlike "*$ExpectedScriptNorm*") {
    $drift += "action_script_mismatch"
  }

  $hasLogon = $false
  foreach ($tr in @($Task.Triggers)) {
    $triggerType = ""
    try { $triggerType = [string]$tr.CimClass.CimClassName } catch { $triggerType = [string]$tr.GetType().Name }
    if ($triggerType -match "Logon") {
      $hasLogon = $true
      break
    }
  }
  if (-not $hasLogon) {
    $drift += "missing_logon_trigger"
  }
  return $drift
}

if ($Apply -and $DryRun) {
  throw "choose_one_mode: use --DryRun or --Apply"
}
if (-not $Apply -and -not $DryRun) {
  $DryRun = $true
}

$repoRoot = Resolve-RepoRoot -ExplicitRoot $Root
if (-not $ManifestPath -or -not $ManifestPath.Trim()) {
  $ManifestPath = Join-Path $repoRoot "config\expected_tasks.json"
}
$manifestFullPath = [System.IO.Path]::GetFullPath($ManifestPath)
$manifest = Get-Content -Raw -Encoding UTF8 $manifestFullPath | ConvertFrom-Json
$expectedTasks = @($manifest.tasks)

$operations = @()
$appliedCount = 0
$wouldChange = $false
$mode = if ($Apply) { "apply" } else { "dry_run" }

foreach ($taskDef in $expectedTasks) {
  $taskNameRaw = [string]$taskDef.task_name
  $split = Split-TaskName -TaskFullName $taskNameRaw
  $taskName = [string]$split.TaskName
  $taskPath = [string]$split.TaskPath
  $scriptRel = [string]$taskDef.action.script
  $scriptPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $scriptRel))
  $scriptNorm = Normalize-PathText $scriptPath
  $scriptArgs = [string]$taskDef.action.args
  $triggerUser = [string]$taskDef.trigger.user
  $triggerDelay = [string]$taskDef.trigger.delay
  $restartCount = [int]$taskDef.settings.restart_count
  $restartInterval = [string]$taskDef.settings.restart_interval
  $hidden = [bool]$taskDef.settings.hidden
  $runLevel = if ([bool]$taskDef.settings.run_with_highest) { "Highest" } else { "LeastPrivilege" }

  $existing = Get-ScheduledTask -TaskName $taskName -TaskPath $taskPath -ErrorAction SilentlyContinue
  $drift = Evaluate-TaskDrift -Task $existing -TaskDef $taskDef -ExpectedScriptNorm $scriptNorm
  $action = if ($null -eq $existing) { "create" } elseif ($drift.Count -gt 0) { "update" } else { "noop" }
  if ($action -ne "noop") {
    $wouldChange = $true
  }

  $op = [ordered]@{
    task_name = $taskNameRaw
    action = $action
    drift = $drift
    applied = $false
    error = $null
  }

  if ($Apply -and $action -ne "noop") {
    try {
      if (-not (Test-Path -LiteralPath $scriptPath)) {
        throw "action_script_missing:$scriptPath"
      }
      $execute = [System.IO.Path]::Combine($PSHOME, "powershell.exe")
      $actionArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`" -Root `"$repoRoot`""
      if ($scriptArgs -and $scriptArgs.Trim()) {
        $actionArgs = "$actionArgs $scriptArgs"
      }
      $taskAction = New-ScheduledTaskAction -Execute $execute -Argument $actionArgs -WorkingDirectory $repoRoot
      $taskTrigger = New-ScheduledTaskTrigger -AtLogOn -User $triggerUser
      if ($triggerDelay -and $triggerDelay.Trim()) {
        try {
          $taskTrigger.Delay = $triggerDelay
        } catch {
        }
      }
      $restartIntervalSpan = Parse-IsoDurationToTimeSpan -IsoDuration $restartInterval
      $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew `
        -Hidden:$hidden `
        -RestartCount $restartCount `
        -RestartInterval $restartIntervalSpan
      $principal = New-ScheduledTaskPrincipal -UserId $triggerUser -LogonType Interactive -RunLevel $runLevel
      Register-ScheduledTask `
        -TaskName $taskName `
        -TaskPath $taskPath `
        -Action $taskAction `
        -Trigger $taskTrigger `
        -Settings $settings `
        -Principal $principal `
        -Description ([string]$taskDef.description) `
        -Force | Out-Null
      $op.applied = $true
      $appliedCount += 1
    } catch {
      $op.error = $_.Exception.Message
    }
  }

  $operations += $op
}

$auditScript = Join-Path $PSScriptRoot "ajax_tasks_audit.ps1"
$auditOutput = & $auditScript -Root $repoRoot -ManifestPath $manifestFullPath
$auditPayload = $null
try {
  $auditPayload = ($auditOutput | Out-String) | ConvertFrom-Json
} catch {
  $auditPayload = [ordered]@{
    ok = $false
    error = "audit_after_parse_failed"
    raw = ($auditOutput | Out-String)
  }
}

$ok = $true
if ($Apply) {
  $missingAfter = @()
  $driftedAfter = @()
  if ($null -ne $auditPayload) {
    if ($auditPayload.PSObject.Properties.Name -contains "missing") { $missingAfter = @($auditPayload.missing) }
    if ($auditPayload.PSObject.Properties.Name -contains "drifted") { $driftedAfter = @($auditPayload.drifted) }
  }
  if ($missingAfter.Count -gt 0 -or $driftedAfter.Count -gt 0) {
    $ok = $false
  }
  if (($operations | Where-Object { $_.error }).Count -gt 0) {
    $ok = $false
  }
}

$payload = [ordered]@{
  schema = "ajax.ops.tasks_ensure.v1"
  ts_utc = (Get-Date).ToUniversalTime().ToString("o")
  ok = $ok
  mode = $mode
  applied = [bool]$Apply
  would_change = $wouldChange
  applied_count = $appliedCount
  repo_root = $repoRoot
  manifest_path = $manifestFullPath
  operations = $operations
  audit_after = $auditPayload
}

$payload | ConvertTo-Json -Depth 20
