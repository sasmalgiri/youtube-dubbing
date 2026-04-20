@echo off
:: VoiceDub Backend — STABLE launcher
::
:: Differences from start.bat:
::   1. Backend runs in THIS visible window (not minimized)
::   2. All output is logged to backend/logs/backend.log
::   3. Auto-restarts on crash with a 5-second delay
::   4. Closing this window is the ONLY way to stop the backend
::   5. No keypress hooks, no taskkill on exit
::
:: To use:  double-click this file (NOT start.bat)
:: To stop: close this window or press Ctrl+C in it

title VoiceDub Backend (STABLE)
color 0B

cd /d "%~dp0backend"

set PYTHONIOENCODING=utf-8
chcp 65001 >nul 2>&1

:: Find Python
set PYTHON=
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set PYTHON=python
    ) else (
        echo [ERROR] Python not found.
        pause
        exit /b 1
    )
)

echo.
echo  ============================================
echo   VoiceDub Backend - Stable Launcher
echo  ============================================
echo  Python:   %PYTHON%
echo  Workdir:  %CD%
echo  Log file: %CD%\logs\backend.log
echo  Port:     8000
echo.
echo  Backend will auto-restart if it crashes.
echo  Press Ctrl+C twice to stop permanently.
echo  ============================================
echo.

:LOOP
echo [%DATE% %TIME%] Starting backend...
%PYTHON% app.py
set EXITCODE=%ERRORLEVEL%
echo.
echo [%DATE% %TIME%] Backend exited with code %EXITCODE%
echo.

if %EXITCODE% EQU 0 (
    echo Backend exited cleanly. Press Ctrl+C to quit, or any key to restart.
    pause >nul
) else (
    echo Backend crashed. Restarting in 5 seconds...
    echo Check logs\backend.log for the traceback.
    timeout /t 5 /nobreak >nul
)
goto LOOP
