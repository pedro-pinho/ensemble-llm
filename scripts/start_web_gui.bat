@echo off
echo Starting Ensemble LLM Web GUI...

REM Check if Ollama is running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if NOT "%ERRORLEVEL%"=="0" (
    echo Starting Ollama service...
    start /B ollama serve
    timeout /t 3 >nul
)

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Please run setup first.
    exit /b 1
)

REM Start the web server
python run_web_gui.py