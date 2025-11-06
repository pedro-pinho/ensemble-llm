#!/bin/bash

# Ensemble LLM Installer Script

set -e

echo "🚀 Ensemble LLM Installer"
echo "========================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
else
    echo -e "${RED}❌ Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

echo "📍 Detected OS: $OS"
echo ""

# Check Python version
echo "🐍 Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Install Ollama
echo ""
echo "🦙 Installing Ollama..."
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✅ Ollama already installed${NC}"
else
    if [[ "$OS" == "macos" ]]; then
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            curl -fsSL https://ollama.ai/install.sh | sh
        fi
    else
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
fi

# Start Ollama service
echo ""
echo "🔧 Starting Ollama service..."
if pgrep -x "ollama" > /dev/null; then
    echo -e "${GREEN}✅ Ollama already running${NC}"
else
    ollama serve > /dev/null 2>&1 &
    sleep 3
    echo -e "${GREEN}✅ Ollama service started${NC}"
fi

# Create virtual environment
echo ""
echo "🏗️  Setting up Python environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate and install dependencies
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Pull models
echo ""
echo "📦 Downloading models (this may take a while)..."
MODELS=("llama3.2:3b" "phi3.5" "qwen2.5:7b" "mistral:7b-instruct-q4_K_M" "gemma2:2b")

for model in "${MODELS[@]}"; do
    echo -n "  Pulling $model... "
    if ollama list | grep -q "$model"; then
        echo -e "${GREEN}already exists${NC}"
    else
        if ollama pull "$model" > /dev/null 2>&1; then
            echo -e "${GREEN}done${NC}"
        else
            echo -e "${YELLOW}failed (optional)${NC}"
        fi
    fi
done

# Run health check
echo ""
echo "🏥 Running health check..."
chmod +x scripts/check_ollama.sh
./scripts/check_ollama.sh

# Create shell alias
echo ""
echo "🔗 Creating shell alias..."
SHELL_RC=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [ ! -z "$SHELL_RC" ]; then
    if ! grep -q "ensemble-llm" "$SHELL_RC"; then
        echo "alias ensemble-llm='cd $(pwd) && source venv/bin/activate && python -m ensemble_llm.main'" >> "$SHELL_RC"
        echo -e "${GREEN}Added 'ensemble-llm' alias to $SHELL_RC${NC}"
        echo -e "${YELLOW}   Run 'source $SHELL_RC' to use the alias${NC}"
    fi
fi

echo ""
echo "Installation complete!"
echo ""
echo "To get started:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Run a query: python -m ensemble_llm.main 'Your question here'"
echo "  3. Or use interactive mode: python -m ensemble_llm.main -i"
echo ""