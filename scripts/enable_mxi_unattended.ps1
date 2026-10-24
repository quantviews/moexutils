param(
    [string]$Python = 'H:\conda\envs\py312\python.exe',
    [string]$Root = 'F:\moex-data\observations\mxi-20261024',
    [Parameter(Mandatory=$true)][string]$RecordDirectory
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$admin = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $admin.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this migration with Windows administrator elevation (UAC).'
}
New-Item -ItemType Directory -Path $RecordDirectory -ErrorAction Stop | Out-Null
$probeName = 'MOEX MXI unattended verification'
$changed = $false
$registeredProbe = $false
try {
    $state = Get-Content -LiteralPath (Join-Path $Root 'start.json') -Raw | ConvertFrom-Json
    $task = Get-ScheduledTask -TaskName $state.task_name
    if ($task.State -eq 'Running') { throw 'Monitor is running; retry migration when idle.' }
    if (Get-ScheduledTask -TaskName $probeName -ErrorAction SilentlyContinue) { throw 'Probe task already exists.' }
    Export-ScheduledTask -TaskName $task.TaskName | Set-Content -LiteralPath (Join-Path $RecordDirectory 'task-before.xml') -Encoding utf8
    $principal = New-ScheduledTaskPrincipal -UserId $task.Principal.UserId -LogonType S4U -RunLevel Limited
    $probeRunner = Join-Path $RecordDirectory 'probe.ps1'
    $probeOutput = Join-Path $RecordDirectory 'probe-result.json'
    # Fixed paths supplied by this migration; no credentials are embedded.
    @"
`$ErrorActionPreference = 'Stop'
`$env:MOEX_DATA_ROOT = 'F:\moex-data'
`$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath '$project'
& '$Python' -m scripts.probe_mxi_unattended --root '$Root' --output '$probeOutput'
exit `$LASTEXITCODE
"@ | Set-Content -LiteralPath $probeRunner -Encoding utf8
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$probeRunner`"" -WorkingDirectory $project
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 4) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $probeName -Action $action -Principal $principal -Settings $settings | Out-Null
    $registeredProbe = $true
    Export-ScheduledTask -TaskName $probeName | Set-Content -LiteralPath (Join-Path $RecordDirectory 'probe-task.xml') -Encoding utf8
    Start-ScheduledTask -TaskName $probeName
    $deadline = (Get-Date).AddMinutes(3)
    do {
        Start-Sleep -Milliseconds 500
        $info = Get-ScheduledTaskInfo -TaskName $probeName
        $running = (Get-ScheduledTask -TaskName $probeName).State -eq 'Running'
    } while (($running -or -not (Test-Path -LiteralPath $probeOutput)) -and (Get-Date) -lt $deadline)
    if (-not (Test-Path -LiteralPath $probeOutput)) { throw 'Probe did not produce a result; original task unchanged.' }
    $probe = Get-Content -LiteralPath $probeOutput -Raw | ConvertFrom-Json
    if ($running -or $info.LastTaskResult -ne 0 -or -not $probe.success) { throw 'S4U dependency probe failed; original task unchanged.' }
    Set-ScheduledTask -TaskName $task.TaskName -Principal $principal | Out-Null
    $changed = $true
    Export-ScheduledTask -TaskName $task.TaskName | Set-Content -LiteralPath (Join-Path $RecordDirectory 'task-after.xml') -Encoding utf8
    $beforeRun = (Get-ScheduledTaskInfo -TaskName $task.TaskName).LastRunTime
    Start-ScheduledTask -TaskName $task.TaskName
    $deadline = (Get-Date).AddMinutes(3)
    do {
        Start-Sleep -Milliseconds 500
        $info = Get-ScheduledTaskInfo -TaskName $task.TaskName
        $running = (Get-ScheduledTask -TaskName $task.TaskName).State -eq 'Running'
    } while (($running -or $info.LastRunTime -le $beforeRun) -and (Get-Date) -lt $deadline)
    if ($running -or $info.LastRunTime -le $beforeRun -or $info.LastTaskResult -ne 0) { throw 'Migrated monitor launch did not succeed.' }
    [pscustomobject]@{Success=$true;TaskName=$task.TaskName;LogonType='S4U';LastRunTime=$info.LastRunTime.ToString('o');LastTaskResult=$info.LastTaskResult;NextRunTime=$info.NextRunTime.ToString('o');CheckedAt=(Get-Date).ToString('o');OriginalEnd=$state.ends_at;Probe=$probeOutput} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RecordDirectory 'migration-result.json') -Encoding utf8
} catch {
    if ($changed) {
        Register-ScheduledTask -TaskName $task.TaskName -Xml (Get-Content -LiteralPath (Join-Path $RecordDirectory 'task-before.xml') -Raw) -Force | Out-Null
    }
    [pscustomobject]@{Success=$false;Error=$_.Exception.Message;OriginalTaskRestored=$changed;CheckedAt=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RecordDirectory 'migration-result.json') -Encoding utf8
    exit 1
} finally {
    if ($registeredProbe -and (Get-ScheduledTask -TaskName $probeName).State -ne 'Running') {
        Unregister-ScheduledTask -TaskName $probeName -Confirm:$false
    }
}
