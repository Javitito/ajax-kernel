param(
  [string]$Root,
  [string]$ManifestPath
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

function To-IsoUtc {
  param([object]$Value)
  if ($null -eq $Value) {
    return $null
  }
  try {
    $dt = [datetime]$Value
  } catch {
    return $null
  }
  if ($dt.Year -lt 2000) {
    return $null
  }
  return $dt.ToUniversalTime().ToString("o")
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

function Normalize-PathText {
  param([string]$Value)
  if (-not $Value) {
    return ""
  }
  return $Value.ToLowerInvariant().Replace("/", "\")
}

$repoRoot = Resolve-RepoRoot -ExplicitRoot $Root
if (-not $ManifestPath -or -not $ManifestPath.Trim()) {
  $ManifestPath = Join-Path $repoRoot "config\expected_tasks.json"
}
$manifestFullPath = [System.IO.Path]::GetFullPath($ManifestPath)
$manifest = Get-Content -Raw -Encoding UTF8 $manifestFullPath | ConvertFrom-Json
$expectedTasks = @($manifest.tasks)

$ts = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$artifactDir = Join-Path $repoRoot "artifacts\audit\tasks_$ts"
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null

$rows = @()
$missing = @()
$drifted = @()
$warnings = @()

foreach ($taskDef in $expectedTasks) {
  $taskNameRaw = [string]$taskDef.task_name
  $split = Split-TaskName -TaskFullName $taskNameRaw
  $taskName = [string]$split.TaskName
  $taskPath = [string]$split.TaskPath
  $expectedUser = [string]$taskDef.trigger.user
  $expectedScriptRel = [string]$taskDef.action.script
  $expectedScript = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $expectedScriptRel))
  $expectedScriptNorm = Normalize-PathText $expectedScript

  $task = Get-ScheduledTask -TaskName $taskName -TaskPath $taskPath -ErrorAction SilentlyContinue
  if ($null -eq $task) {
    $missing += $taskNameRaw
    $rows += [ordered]@{
      task_name = $taskNameRaw
      exists = $false
      enabled = $false
      state = "Missing"
      last_run_result = $null
      last_run_time = $null
      next_run_time = $null
      run_as_user = $null
      run_level = $null
      actions = @()
      triggers = @()
      drift = @("missing_task")
    }
    continue
  }

  $info = Get-ScheduledTaskInfo -TaskName $taskName -TaskPath $taskPath -ErrorAction SilentlyContinue
  $actions = @()
  foreach ($act in @($task.Actions)) {
    $actions += [ordered]@{
      execute = [string]$act.Execute
      arguments = [string]$act.Arguments
      working_directory = [string]$act.WorkingDirectory
    }
  }
  $triggers = @()
  foreach ($tr in @($task.Triggers)) {
    $triggerType = $null
    try {
      $triggerType = [string]$tr.CimClass.CimClassName
    } catch {
      $triggerType = [string]$tr.GetType().Name
    }
    $triggers += [ordered]@{
      type = $triggerType
      user = [string]$tr.UserId
      enabled = [bool]$tr.Enabled
      delay = [string]$tr.Delay
      start_boundary = [string]$tr.StartBoundary
    }
  }

  $drift = @()
  $enabled = $true
  try {
    $enabled = [bool]$task.Settings.Enabled
  } catch {
    $enabled = $true
  }
  if (-not $enabled) {
    $drift += "disabled"
  }
  $runLevel = [string]$task.Principal.RunLevel
  if ([string]$taskDef.settings.run_with_highest -eq "True" -and $runLevel -notmatch "Highest") {
    $drift += "run_level_not_highest"
  }
  $runAs = [string]$task.Principal.UserId
  if ($expectedUser -and $runAs -and $runAs.ToLowerInvariant() -notlike "*$($expectedUser.ToLowerInvariant())*") {
    $drift += "run_as_user_mismatch"
  }

  $actionBlob = ""
  if ($actions.Count -gt 0) {
    $actionBlob = ([string]$actions[0].execute + " " + [string]$actions[0].arguments).Trim()
  }
  $actionBlobNorm = Normalize-PathText $actionBlob
  if ($expectedScriptNorm -and $actionBlobNorm -notlike "*$expectedScriptNorm*") {
    $drift += "action_script_mismatch"
  }

  $hasLogon = $false
  $hasExpectedUser = $false
  foreach ($tr in $triggers) {
    if ([string]$tr.type -match "Logon") {
      $hasLogon = $true
      if (-not $expectedUser -or [string]$tr.user -eq "" -or [string]$tr.user -match $expectedUser) {
        $hasExpectedUser = $true
      }
    }
  }
  if (-not $hasLogon) {
    $drift += "missing_logon_trigger"
  } elseif (-not $hasExpectedUser) {
    $drift += "logon_user_mismatch"
  }

  if ($drift.Count -gt 0) {
    $drifted += $taskNameRaw
  }

  $safeName = ($taskNameRaw.TrimStart("\").Replace("\", "__"))
  try {
    $xml = Export-ScheduledTask -TaskName $taskName -TaskPath $taskPath -ErrorAction Stop
    $xmlPath = Join-Path $artifactDir "$safeName.xml"
    Set-Content -Path $xmlPath -Value $xml -Encoding UTF8
  } catch {
    $warnings += [ordered]@{
      task_name = $taskNameRaw
      code = "export_xml_failed"
      detail = $_.Exception.Message
    }
  }

  $lastTaskResult = $null
  $lastRunTime = $null
  $nextRunTime = $null
  if ($null -ne $info) {
    try { $lastTaskResult = [int]$info.LastTaskResult } catch { $lastTaskResult = $null }
    $lastRunTime = To-IsoUtc $info.LastRunTime
    $nextRunTime = To-IsoUtc $info.NextRunTime
  }

  $rows += [ordered]@{
    task_name = $taskNameRaw
    exists = $true
    enabled = $enabled
    state = [string]$task.State
    last_run_result = $lastTaskResult
    last_run_result_hex = if ($null -ne $lastTaskResult) { ("0x{0:X8}" -f ([uint32]$lastTaskResult)) } else { $null }
    last_run_time = $lastRunTime
    next_run_time = $nextRunTime
    run_as_user = $runAs
    run_level = $runLevel
    actions = $actions
    triggers = $triggers
    drift = $drift
  }
}

$payload = [ordered]@{
  schema = "ajax.ops.tasks_audit.v1"
  ts_utc = (Get-Date).ToUniversalTime().ToString("o")
  ok = $true
  repo_root = $repoRoot
  manifest_path = $manifestFullPath
  artifact_dir = $artifactDir
  tasks = $rows
  missing = ($missing | Sort-Object -Unique)
  drifted = ($drifted | Sort-Object -Unique)
  warnings = $warnings
}

$reportPath = Join-Path $artifactDir "tasks_audit.json"
$payload | ConvertTo-Json -Depth 16 | Set-Content -Path $reportPath -Encoding UTF8
$payload | ConvertTo-Json -Depth 16
