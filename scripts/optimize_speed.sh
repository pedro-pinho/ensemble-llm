#!/bin/bash

echo "Optimizing Ollama for speed..."

# Set Ollama environment variables for speed
export OLLAMA_NUM_PARALLEL=2  # Limit parallel requests
export OLLAMA_MAX_LOADED_MODELS=2  # Only keep 2 models in memory
export OLLAMA_MODELS_DIR=${OLLAMA_MODELS_DIR:-~/.ollama/models}
export OLLAMA_KEEP_ALIVE=5m  # Keep models loaded for 5 minutes

# Kill existing Ollama processes
echo "Restarting Ollama service..."
killall ollama 2>/dev/null || true
sleep 2

# Start Ollama with optimized settings
nohup ollama serve > /dev/null 2>&1 &
sleep 3

# Load only fast models
echo "Loading fast models..."
ollama run tinyllama:1b "Hi" > /dev/null 2>&1 &
ollama run gemma2:2b "Hi" > /dev/null 2>&1 &
wait

echo "Speed optimization complete!"
echo ""
echo "Recommended usage for maximum speed:"
echo "  python -m ensemble_llm.main --speed turbo 'Your question'"
echo ""
echo "Or for better quality with good speed:"
echo "  python -m ensemble_llm.main --speed fast 'Your question'"