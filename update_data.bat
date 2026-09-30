@echo off
rem Update local MOEX data: stocks, indexes, bonds, futures, other markets, rates, cash flows,
rem security parameters, quality, backup.
rem Run by double click. Flags are passed to update_data.py,
rem e.g.: update_data.bat --no-adj --no-cap
rem Interpreter: MOEX_PYTHON env var, else conda env py312, else python from PATH
rem (the system python lacks moexutils, polars, duckdb). MOEX_NO_PAUSE=1 disables the final pause
rem (set by scheduled_update.cmd for Task Scheduler).
rem The file is ASCII-only on purpose: cmd misparses UTF-8 batch files after chcp 65001.
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"

set "PY=%MOEX_PYTHON%"
if not defined PY if exist "H:\conda\envs\py312\python.exe" set "PY=H:\conda\envs\py312\python.exe"
if not defined PY set "PY=python"

"%PY%" -c "import moexutils, polars, duckdb" 2>nul
if errorlevel 1 (
    echo [ERROR] Python "%PY%" cannot import moexutils/polars/duckdb (pip install -e . in the project folder).
    echo Set MOEX_PYTHON to the interpreter of the project environment.
    set "RC=1"
    goto :end
)

"%PY%" update_data.py %*
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo.
    echo [ERROR] Update failed, exit code %RC%.
)

:end
if not defined MOEX_NO_PAUSE pause
exit /b %RC%
