# Ensemble LLM

A powerful local LLM ensemble system that runs multiple language models in parallel and uses voting mechanisms to provide the best possible answers. Features include multi-model consensus, web search integration, and intelligent response selection.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

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


## Windows Installation and Setup 🪟

### Prerequisites

1. **Python 3.8+**
   - Download from [python.org](https://python.org)
   - ✅ **Important**: Check "Add Python to PATH" during installation

2. **Ollama for Windows**
   - Download from [ollama.ai/download/windows](https://ollama.ai/download/windows)
   - Run the installer (requires Windows 10/11)

3. **Git for Windows** (optional but recommended)
   - Download from [git-scm.com](https://git-scm.com)

4. **NVIDIA GPU** (optional but recommended)
   - Install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for GPU acceleration
   - Ensure you have the latest NVIDIA drivers

### Quick Setup (Windows)

1. **Clone or download the repository**
```powershell
# Using Git
git clone https://github.com/yourusername/ensemble-llm.git
cd ensemble-llm

# Or download and extract the ZIP file from GitHub
```

2. **Run the Windows setup script**
```powershell
# In PowerShell or Command Prompt
scripts\setup_windows.bat
```

This will:
- Check Python and Ollama installation
- Create virtual environment
- Install all dependencies
- Create necessary directories

3. **Start Ollama with GPU optimization**
```powershell
scripts\start_ollama_windows.bat
```

4. **Pull models**
```powershell
# Activate virtual environment
venv\Scripts\activate

# Pull recommended models for GPU
ollama pull llama3.2:3b
ollama pull phi3.5
ollama pull qwen2.5:7b
ollama pull mistral:7b-instruct-q4_K_M
```

### Running on Windows

#### Basic Usage
```powershell
# Activate virtual environment
venv\Scripts\activate

# Run a query
python -m ensemble_llm.main "Your question here"

# With GPU optimization
set OLLAMA_NUM_GPU=999
python -m ensemble_llm.main --speed fast "Your question"

# Interactive mode
python -m ensemble_llm.main -i
```

#### PowerShell Alias (Optional)
Add to your PowerShell profile (`$PROFILE`):
```powershell
function ensemble {
    & "$env:USERPROFILE\ensemble-llm\venv\Scripts\python.exe" -m ensemble_llm.main $args
}
```

Then use: `ensemble "Your question"`

### GPU Optimization (Windows)

#### Check GPU Status
```powershell
# Run the benchmark script
python scripts\benchmark_windows.py
```

#### Environment Variables for GPU
```powershell
# Set these before running for maximum GPU usage
set OLLAMA_NUM_GPU=999          # Use all GPU layers
set CUDA_VISIBLE_DEVICES=0      # Use first GPU (or 0,1 for multi-GPU)
set OLLAMA_MAX_LOADED_MODELS=4  # Load more models with GPU memory
set OLLAMA_KEEP_ALIVE=10m       # Keep models in VRAM longer
```

#### Recommended Configurations by GPU

| GPU VRAM | Recommended Models | Max Concurrent |
|----------|-------------------|----------------|
| 6GB | `tinyllama:1b`, `gemma2:2b`, `llama3.2:3b` | 3 |
| 8GB | `llama3.2:3b`, `phi3.5`, `qwen2.5:7b`, `mistral:7b` | 4 |
| 12GB | `llama3.1:8b`, `qwen2.5:7b`, `mistral:7b`, `codellama:7b` | 4 |
| 16GB | `llama3.1:13b`, `mixtral:8x7b-q3`, `qwen2.5:7b` | 3 |
| 24GB | `llama3.1:70b-q2`, `mixtral:8x7b`, `llama3.1:13b` | 2-3 |

### Windows-Specific Features

#### 1. Process Priority Optimization
```python
# The system automatically sets Ollama to high priority on Windows
python -m ensemble_llm.main --optimize-windows "Your question"
```

#### 2. GPU Memory Monitoring
```powershell
# Monitor GPU usage while running
nvidia-smi -l 1

# In another terminal
python -m ensemble_llm.main -i --speed turbo
```

#### 3. Windows Task Scheduler (Auto-warmup)
Create a scheduled task to warmup models on system start:
```powershell
# Create warmup script: warmup.bat
@echo off
cd C:\path\to\ensemble-llm
call venv\Scripts\activate
python -c "from ensemble_llm.fast_mode import ModelWarmup; import asyncio; mw = ModelWarmup(); asyncio.run(mw.parallel_warmup(['llama3.2:3b', 'phi3.5:latest']))"
```

### Troubleshooting (Windows)

#### Ollama not found
```powershell
# Add Ollama to PATH manually
set PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama

# Or reinstall Ollama and ensure "Add to PATH" is checked
```

#### GPU not detected
```powershell
# Check CUDA installation
nvidia-smi

# Check if Ollama detects GPU
ollama run llama3.2:3b --verbose

# Should show: "Loading model on GPU..."
```

#### Permission errors
```powershell
# Run PowerShell as Administrator
# Or ensure your user has write permissions to the project directory
```

#### Memory errors with multiple models
```powershell
# Reduce number of concurrent models
python -m ensemble_llm.main --models llama3.2:3b gemma2:2b --speed fast "Query"

# Or use smaller quantized versions
ollama pull llama3.2:1b  # Smaller version
```

### Windows Performance Tips

1. **Use GPU-optimized models**: Models with Q4_K_M quantization work best on GPUs
2. **Close unnecessary programs**: Free up RAM and VRAM
3. **Use Windows GPU scheduling**: Settings → System → Display → Graphics settings → Hardware-accelerated GPU scheduling
4. **Disable Windows Defender scanning** for the Ollama models directory (with caution)
5. **Use NVMe SSD** for model storage if possible

### Example: Maximum Performance on Windows Gaming PC
```powershell
# For a system with RTX 4090 (24GB VRAM), 64GB RAM

# Set environment
set OLLAMA_NUM_GPU=999
set OLLAMA_MAX_LOADED_MODELS=6
set CUDA_VISIBLE_DEVICES=0

# Start Ollama
scripts\start_ollama_windows.bat

# Use large models
python -m ensemble_llm.main ^
  --models llama3.1:13b mixtral:8x7b qwen2.5:14b ^
  --speed balanced ^
  "Complex question requiring deep reasoning"
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
