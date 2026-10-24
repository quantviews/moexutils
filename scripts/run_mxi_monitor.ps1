param(
    [string]$Python = 'H:\conda\envs\py312\python.exe',
    [string]$Root = 'F:\moex-data\observations\mxi-20261024'
)
$ErrorActionPreference = 'Stop'
$env:MOEX_DATA_ROOT = 'F:\moex-data'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
& $Python -m moexutils.mxi_monitor tick --root $Root
exit $LASTEXITCODE
