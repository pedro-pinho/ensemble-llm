#!/bin/bash
# check_ollama.sh

echo "🔍 Checking Ollama Setup..."
echo "================================"

# Check if Ollama is running
if pgrep -x "ollama" > /dev/null; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not running. Starting..."
    ollama serve &
    sleep 3
fi

# List available models
echo -e "\n📦 Available Models:"
ollama list

# Check each model
echo -e "\n🧪 Testing Models:"
for model in llama3.2:3b phi3.5:latest qwen2.5:7b mistral:7b-instruct-q4_K_M gemma2:2b; do
    echo -n "Testing $model... "
    if ollama run $model "Hi" > /dev/null 2>&1; then
        echo "✅ OK"
    else
        echo "❌ Failed"
        echo "  Try: ollama pull $model"
    fi
done

echo -e "\n💾 Memory Usage:"
ps aux | grep ollama | awk '{sum+=$6} END {print "Ollama is using " sum/1024 " MB of RAM"}'

echo -e "\n🌐 API Status:"
curl -s http://localhost:11434/api/tags > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ API is accessible at http://localhost:11434"
else
    echo "❌ API is not accessible"
fi