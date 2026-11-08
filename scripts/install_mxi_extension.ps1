$ErrorActionPreference = 'Stop'
$name = 'MOEX MXI morning delivery 20270226'
$project = Split-Path -Parent $PSScriptRoot
if ((Get-TimeZone).Id -ne 'Russian Standard Time') { throw 'Moscow timezone required' }
if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { throw 'Task already exists' }
$original = Get-ScheduledTask -TaskName 'MOEX MXI observations 30d'
$principal = New-ScheduledTaskPrincipal -UserId $original.Principal.UserId -RunLevel Limited -LogonType S4U
$triggers = foreach ($minute in @(0,30)) {
    $t = New-ScheduledTaskTrigger -Daily -At ([datetime]::new(2026,11,24,9,$minute,0))
    $t.EndBoundary = '2027-02-26T23:59:59+03:00'
    $t
}
$runner = Join-Path $PSScriptRoot 'run_mxi_extension.ps1'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $project -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$runner`""
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $name -Action $action -Trigger $triggers -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $name
