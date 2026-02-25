@echo off
title VoiceDub Frontend
echo ========================================
echo   VoiceDub - Starting Frontend...
echo ========================================
echo.

:: Kill any existing node processes
taskkill /F /IM node.exe >nul 2>&1

:: Clean corrupted cache
if exist "web\.next" (
    echo Cleaning build cache...
    rmdir /s /q "web\.next"
)

:: Start dev server
echo Starting Next.js on http://localhost:3000
echo.
cd web
npm run dev
