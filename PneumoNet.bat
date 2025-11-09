@echo off
title 🫁 PneumoNet - Pneumonia Detection System (FastAPI + Vue)
echo ============================================================
echo 🚀 Launching PneumoNet - Pneumonia Detection System
echo ============================================================

REM --- Automatically detect this script's directory ---
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
echo Working directory: %CD%

REM --- Activate Python virtual environment ---
echo.
echo [1/4] Activating Python virtual environment...
if exist "venv\Scripts\activate" (
    call venv\Scripts\activate
) else (
    echo ❌ Virtual environment not found at venv\Scripts\activate
    echo Please check your venv path or recreate it using:
    echo     python -m venv venv
    pause
    exit /b
)

REM --- Start FastAPI backend ---
echo.
echo [2/4] Starting FastAPI backend...
if exist "web\backend\main.py" (
    pushd "web\backend"
    start cmd /k "uvicorn main:app --reload --host 127.0.0.1 --port 8000"
    popd
) else (
    echo ❌ Could not find main.py in web\backend
    pause
    exit /b
)

REM --- Start Vue frontend ---
echo.
echo [3/4] Starting Vue frontend...
if exist "web\frontend\package.json" (
    pushd "web\frontend"
    if not exist node_modules (
        echo Installing npm dependencies...
        call npm install
    )
    start cmd /k "npm run dev"
    popd
) else (
    echo ❌ Could not find web\frontend\package.json
    pause
    exit /b
)

REM --- Open both URLs automatically ---
echo.
echo [4/4] Opening browser...
start "" "http://127.0.0.1:8000/docs"
start "" "http://localhost:5173/"

echo.
echo ============================================================
echo ✅ PneumoNet is now running!
echo - FastAPI: http://127.0.0.1:8000/docs
echo - Vue:     http://localhost:5173/
echo ============================================================
pause
