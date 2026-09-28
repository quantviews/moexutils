@echo off
rem Nightly data update for Windows Task Scheduler (task "MOEX data nightly").
rem No interactive pause; output is appended to logs\update.log,
rem which is rotated to logs\update.old.log after 5 MB.
cd /d "%~dp0"
if not exist logs mkdir logs
for %%F in (logs\update.log) do if %%~zF GTR 5000000 move /y logs\update.log logs\update.old.log >nul
set "MOEX_NO_PAUSE=1"
echo ===== start %date% %time% >> logs\update.log
call "%~dp0update_data.bat" %* >> logs\update.log 2>&1
set "RC=%errorlevel%"
echo ===== exit %RC% %date% %time% >> logs\update.log
exit /b %RC%
