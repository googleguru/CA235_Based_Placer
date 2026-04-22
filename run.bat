@echo off
title DREAMPlace MetaOpt — VLSI Placement
echo ============================================================
echo   DREAMPlace MetaOpt - Metaheuristic VLSI Placement
echo   No Deep Learning - Pure Optimization
echo ============================================================
echo.

cd /d "%~dp0"

:: Check for Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.8+ first.
    echo         https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Run the main script (GUI mode by default)
echo [Starting] Launching placement optimizer with GUI...
echo.
python run.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Something went wrong. Check the output above.
    pause
)
