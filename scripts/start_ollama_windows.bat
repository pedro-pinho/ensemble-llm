@echo off
echo Starting Ollama for Windows with GPU optimization...
echo.

REM Kill existing Ollama process
taskkill /F /IM ollama.exe >nul 2>&1

REM Set environment variables for GPU
set OLLAMA_NUM_GPU=999
set CUDA_VISIBLE_DEVICES=0
set OLLAMA_MAX_LOADED_MODELS=4
set OLLAMA_NUM_PARALLEL=2
set OLLAMA_KEEP_ALIVE=10m

REM Start Ollama
echo Starting Ollama service...
start /B ollama serve

timeout /t 3 >nul

REM Check if running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] Ollama is running
) else (
    echo [ERROR] Failed to start Ollama
    pause
    exit /b 1
)

echo.
echo Ollama started with GPU optimization
echo.
echo Environment:
echo   OLLAMA_NUM_GPU=%OLLAMA_NUM_GPU%
echo   OLLAMA_MAX_LOADED_MODELS=%OLLAMA_MAX_LOADED_MODELS%
echo.
pause