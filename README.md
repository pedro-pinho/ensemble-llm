# Ensemble LLM

A powerful local LLM ensemble system that runs multiple language models in parallel and uses voting mechanisms to provide the best possible answers. Features include multi-model consensus, web search integration, and intelligent response selection.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20|%20Linux-lightgrey.svg)

## Features

- **Multi-Model Ensemble**: Run multiple LLMs simultaneously and get the best answer through intelligent voting
- **Web Search Integration**: Automatically search the web for current information when needed
- **Smart Consensus**: Advanced voting system based on response similarity, quality, and confidence
- **Local Execution**: Everything runs locally on your machine - no API keys required
- **Optimized for Apple Silicon**: Fully utilizes M1/M2 Mac capabilities
- **Extensive Logging**: Detailed logs for debugging and monitoring
- **Interactive Mode**: CLI interface for continuous conversations

## Requirements

- macOS (Apple Silicon recommended) or Linux
- Python 3.8+
- 16GB+ RAM (24GB recommended for running 4+ models)
- 20GB+ free disk space for models
- Ollama installed

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/pedropinho/ensemble-llm.git
cd ensemble-llm
```

### 2. Run the installer
```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

This will:
- Install Ollama (if not already installed)
- Set up Python virtual environment
- Install all dependencies
- Download recommended models
- Run health checks

### 3. Run your first query
```bash
# Activate the environment
source venv/bin/activate

# Simple query
python -m ensemble_llm.main "What is quantum computing?"

# With web search for current events
python -m ensemble_llm.main -w "What's the latest news about AI?"

# Interactive mode
python -m ensemble_llm.main -i -w
```

## Installation (Manual)

### Step 1: Install Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 2: Start Ollama and pull models
```bash
# Start Ollama service
ollama serve

# Pull recommended models (in another terminal)
ollama pull llama3.2:3b
ollama pull phi3.5:latest
ollama pull qwen2.5:7b
ollama pull mistral:7b-instruct-q4_K_M
ollama pull gemma2:2b
```

### Step 3: Set up Python environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Verify installation
```bash
# Check Ollama setup
./scripts/check_ollama.sh

# Test ensemble system
python -m ensemble_llm.main "Hello, how are you?"
```

## Usage

### Basic Usage
```bash
# Simple query with default models
python -m ensemble_llm.main "Explain machine learning"

# Verbose mode - see all model responses
python -m ensemble_llm.main -v "What is the meaning of life?"

# Enable web search for current information
python -m ensemble_llm.main -w "Current stock price of Apple"

# Interactive chat mode
python -m ensemble_llm.main -i
```

### Advanced Usage
```bash
# Use specific models
python -m ensemble_llm.main --models llama3.2:3b phi3.5:latest "Quick question"

# Debug mode with detailed logging
python -m ensemble_llm.main --log-level DEBUG -v "Test query"

# Combine all features
python -m ensemble_llm.main -i -w -v --models llama3.2:3b qwen2.5:7b mistral:7b
```

### Python API
```python
from ensemble_llm import EnsembleLLM
import asyncio

async def main():
    # Initialize ensemble with custom models
    ensemble = EnsembleLLM(
        models=['llama3.2:3b', 'phi3.5:latest', 'qwen2.5:7b'],
        enable_web_search=True
    )
    
    # Get ensemble response
    response, metadata = await ensemble.ensemble_query(
        "What is the future of AI?",
        verbose=True
    )
    
    print(f"Best answer from {metadata['selected_model']}:")
    print(response)
    print(f"Consensus score: {metadata['consensus_score']:.2f}")
    
    # Cleanup
    await ensemble.cleanup()

asyncio.run(main())
```

## Model Recommendations

Based on your available RAM:

### 16GB RAM
- Run 3-4 small models (3B parameters)
- Recommended: `llama3.2:3b`, `phi3.5:latest`, `gemma2:2b`

### 24GB RAM (Optimal)
- Run 4-5 medium models (3-7B parameters)
- Recommended: `llama3.2:3b`, `phi3.5:latest`, `qwen2.5:7b`, `mistral:7b`, `gemma2:2b`

### 32GB+ RAM
- Run 3-4 large models or 6+ small models
- Add: `mixtral:8x7b-instruct-q3_K_M`, `llama3.1:70b-instruct-q2_K`

See [docs/MODELS.md](docs/MODELS.md) for detailed model information.

## How It Works

1. **Query Distribution**: Your question is sent to multiple LLMs in parallel
2. **Response Generation**: Each model generates its answer independently
3. **Web Search** (optional): If current information is needed, web search results are added to the context
4. **Similarity Analysis**: Responses are compared using TF-IDF vectorization
5. **Consensus Scoring**: Each response gets a consensus score based on agreement with other models
6. **Quality Scoring**: Responses are evaluated for length, structure, and detail
7. **Final Selection**: The response with the highest combined score is selected

## Configuration

Edit `ensemble_llm/config.py` to customize:

- Model configurations and specialties
- Voting weights (consensus vs quality)
- Web search settings
- Retry logic and timeouts

## Troubleshooting

### Models not responding
```bash
# Check Ollama service
./scripts/check_ollama.sh

# Restart Ollama
killall ollama
ollama serve
```

### Out of memory errors
- Reduce number of concurrent models
- Use smaller/more quantized models
- Close other applications

### Slow responses
- Check GPU acceleration: `ollama list`
- Reduce model size or number
- Increase timeout values in config

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- [Ollama](https://ollama.ai/) for making local LLM deployment easy
- The open-source AI community for providing amazing models

---