#!/bin/bash

echo "Starting Ensemble LLM Web GUI..."

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
else
    echo "Virtual environment not found. Please run setup first."
    exit 1
fi

# Start the web server
python run_web_gui.py