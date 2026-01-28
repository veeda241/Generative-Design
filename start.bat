@echo off
REM AETHER-GEN Complete Startup Script for Windows
REM This script starts both the backend and frontend servers

echo.
echo ========================================
echo AETHER-GEN - Generative Design System
echo Version 1.5.0 - COMPLETE
echo ========================================
echo.

REM Check if Python environment exists
if not exist ".venv" (
    echo [ERROR] Virtual environment not found. Running setup...
    cd backend
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
    cd ..
    call .venv\Scripts\activate.bat
)

REM Setup complete - start both services
echo.
echo [1] Starting Python Backend (FastAPI) on http://localhost:8000
echo [2] Starting React Frontend on http://localhost:5173
echo.
echo Opening terminals...
echo.

REM Start Backend in new terminal
start "Backend - AETHER-GEN" cmd /k "cd backend && ..\\.venv\\Scripts\\python.exe main.py"

timeout /t 2

REM Start Frontend in new terminal
start "Frontend - AETHER-GEN" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo Services Starting...
echo ========================================
echo.
echo Frontend:  http://localhost:5173
echo Backend:   http://localhost:8000
echo Docs:      http://localhost:8000/docs
echo.
echo Press Ctrl+C in either terminal to stop that service
echo.
pause
