@echo off
rem Обновление локальных данных MOEX: котировки акций в data/, adj_close, market_cap.
rem Запуск: двойной клик или из планировщика. Флаги пробрасываются в update_data.py,
rem например: update_data.bat --no-adj --no-cap
chcp 65001 >nul
cd /d "%~dp0"

python update_data.py %*

if errorlevel 1 (
    echo.
    echo [ОШИБКА] Обновление завершилось с ошибкой, код %errorlevel%.
)
pause
