param([Parameter(Mandatory=$true)][string]$RecordDirectory)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$name = 'MOEX data nightly'
$probeName = 'MOEX nightly S4U verification'
New-Item -ItemType Directory -Path $RecordDirectory -ErrorAction Stop | Out-Null
$changed = $false
$probeRegistered = $false
try {
    $task = Get-ScheduledTask -TaskName $name
    if ($task.State -eq 'Running') { throw 'Nightly update is running; retry when idle.' }
    if (Get-ScheduledTask -TaskName $probeName -ErrorAction SilentlyContinue) { throw 'Probe task already exists.' }
    Export-ScheduledTask -TaskName $name | Set-Content (Join-Path $RecordDirectory 'before.xml') -Encoding utf8
    $principal = New-ScheduledTaskPrincipal -UserId $task.Principal.UserId -LogonType S4U -RunLevel Limited
    $runner = Join-Path $RecordDirectory 'probe.ps1'
    @"
`$ErrorActionPreference = 'Stop'
`$env:MOEX_DATA_ROOT = 'F:\moex-data'
Set-Location -LiteralPath '$project'
& '$project\scheduled_update.cmd' --check
exit `$LASTEXITCODE
"@ | Set-Content $runner -Encoding utf8
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $project -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$runner`""
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $probeName -Action $action -Principal $principal -Settings $settings | Out-Null
    $probeRegistered = $true
    Export-ScheduledTask -TaskName $probeName | Set-Content (Join-Path $RecordDirectory 'probe.xml') -Encoding utf8
    Start-ScheduledTask -TaskName $probeName
    $deadline = (Get-Date).AddMinutes(6)
    do {
        Start-Sleep -Seconds 1
        $info = Get-ScheduledTaskInfo -TaskName $probeName
        $running = (Get-ScheduledTask -TaskName $probeName).State -eq 'Running'
    } while (($running -or $info.LastRunTime.Year -lt 2026) -and (Get-Date) -lt $deadline)
    if ($running -or $info.LastTaskResult -ne 0) { throw 'S4U check failed; nightly task unchanged.' }
    Set-ScheduledTask -TaskName $name -Principal $principal | Out-Null
    $changed = $true
    Export-ScheduledTask -TaskName $name | Set-Content (Join-Path $RecordDirectory 'after.xml') -Encoding utf8
    [pscustomobject]@{Success=$true;Task=$name;LogonType='S4U';ProbeMode='scheduled_update.cmd --check';ProbeRun=$info.LastRunTime.ToString('o');ProbeExitCode=$info.LastTaskResult;CheckedAt=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content (Join-Path $RecordDirectory 'result.json') -Encoding utf8
} catch {
    if ($changed) { Register-ScheduledTask -TaskName $name -Xml (Get-Content (Join-Path $RecordDirectory 'before.xml') -Raw) -Force | Out-Null }
    [pscustomobject]@{Success=$false;Error=$_.Exception.Message} | ConvertTo-Json | Set-Content (Join-Path $RecordDirectory 'result.json') -Encoding utf8
    exit 1
} finally {
    if ($probeRegistered -and (Get-ScheduledTask -TaskName $probeName).State -ne 'Running') {
        Unregister-ScheduledTask -TaskName $probeName -Confirm:$false
    }
}
