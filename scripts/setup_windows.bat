@echo off
echo ===============================================
echo Ensemble LLM - Windows Setup Script
echo ===============================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [OK] Python found
python --version

REM Check if Ollama is installed
echo.
echo Checking for Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama not found in PATH
    echo.
    echo Please install Ollama from: https://ollama.ai/download/windows
    echo.
    set /p CONTINUE=Do you want to continue anyway? (y/n): 
    if /i not "%CONTINUE%"=="y" exit /b 1
) else (
    echo [OK] Ollama found
    ollama --version
)

REM Create virtual environment
echo.
echo Creating Python virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install requirements
echo.
echo Installing requirements...
pip install -r requirements.txt

REM Create necessary directories
echo.
echo Creating directories...
if not exist logs mkdir logs
if not exist data mkdir data
if not exist cache mkdir cache
if not exist smart_data mkdir smart_data

echo.
echo ===============================================
echo Setup complete!
echo ===============================================
echo.
echo To use the ensemble:
echo   1. Run: venv\Scripts\activate
echo   2. Then: python -m ensemble_llm.main "Your question"
echo.
pause