$ErrorActionPreference = 'Stop'
$env:MOEX_DATA_ROOT = 'F:\moex-data'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
& 'H:\conda\envs\py312\python.exe' -m moexutils.mxi_extension
exit $LASTEXITCODE
