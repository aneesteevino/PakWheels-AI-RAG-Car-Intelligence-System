@echo off
SETLOCAL

SET SCRIPT_DIR=%~dp0
SET BACKEND_DIR=%SCRIPT_DIR%backend
SET VENV_DIR=%SCRIPT_DIR%.venv

echo.
echo ╔══════════════════════════════════════════╗
echo ║      PakWheels AI — RAG System           ║
echo ╚══════════════════════════════════════════╝
echo.

IF NOT EXIST "%VENV_DIR%" (
    echo [1/4] Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

echo [2/4] Activating environment...
CALL "%VENV_DIR%\Scripts\activate.bat"

echo [3/4] Installing dependencies...
pip install -q --upgrade pip
pip install -q -r "%BACKEND_DIR%\requirements.txt"

echo [4/4] Starting server on http://localhost:8000
echo.

if not defined GROQ_API_KEY (
    echo ERROR: GROQ_API_KEY environment variable is not set.
    echo Please set it before running:
    echo   set GROQ_API_KEY=your_groq_api_key_here
    pause
    exit /b 1
)

cd /d "%BACKEND_DIR%"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
