param(
    [string]$Python = 'H:\conda\envs\py312\python.exe',
    [string]$Root = 'F:\moex-data\observations\mxi-20261024',
    [string]$Project = 'F:\Yandex.Disk\FINANCE\moexutils'
)
$ErrorActionPreference = 'Stop'
$state = Get-Content -LiteralPath (Join-Path $Root 'start.json') -Raw | ConvertFrom-Json
if ((Get-TimeZone).Id -ne 'Russian Standard Time') { throw 'This task definition requires Moscow Windows timezone' }
$taskName = $state.task_name
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw 'Task already exists; inspect it instead of restarting the observation period'
}
$end = [DateTimeOffset]::Parse($state.ends_at).LocalDateTime
if ($end -le (Get-Date)) { throw 'Observation period has ended' }
$first = (Get-Date).AddMinutes(1)
$first = $first.Date.AddHours($first.Hour).AddMinutes([math]::Ceiling($first.Minute / 5.0) * 5)
$duration = $end.AddMinutes(10) - $first
$trigger = New-ScheduledTaskTrigger -Once -At $first -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration $duration
$trigger.EndBoundary = $end.AddMinutes(10).ToString('yyyy-MM-ddTHH:mm:ss')
$runner = Join-Path $Project 'scripts/run_mxi_monitor.ps1'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$runner`" -Python `"$Python`" -Root `"$Root`"" -WorkingDirectory $Project
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 4) -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Bounded 30-day MXI observations; no retries, 425 position requests; forward inputs fail closed.'
try {
    Register-ScheduledTask -TaskName $taskName -InputObject $task | Out-Null
} catch {
    if ($_.Exception.HResult -ne -2147024891 -and $_.ToString() -notmatch 'Access is denied|0x80070005') { throw }
    # Non-admin accounts may only register tasks for their existing login session.
    $task.Principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -InputObject $task | Out-Null
    Write-Warning 'Installed Interactive: user must remain signed in (screen may be locked).'
}
Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath (Join-Path $Root 'task.xml') -Encoding utf8
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
