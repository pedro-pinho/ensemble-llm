# Ensemble LLM

> A powerful local LLM ensemble system that combines multiple AI models through intelligent voting and synthesis to deliver superior answers.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Ensemble LLM runs multiple local language models in parallel and uses advanced voting algorithms to select or synthesize the best possible answer. Unlike single-model systems, it leverages diverse AI perspectives to provide more comprehensive, balanced, and accurate responses.

### Key Features

**Multi-Model Consensus**
- Run 3-5 models simultaneously through Ollama
- Intelligent voting based on response similarity and quality
- Automatic model selection and rotation

**Council & Synthesis Mode**
- Models collaborate as an AI council with role awareness
- Winning model synthesizes all perspectives into one coherent answer
- Automatic filtering of AI meta-talk for clean, professional outputs

**Web Search Integration**
- Automatically detects when queries need current information
- Enhances prompts with real-time web search results
- Seamless fallback for up-to-date knowledge

**Intelligent Learning System**
- Query caching with similarity matching
- Pattern recognition for query optimization
- Adaptive model performance tracking
- Precomputation of common queries

**Speed Optimization**
- Multiple speed modes: turbo, fast, balanced, quality
- Model warmup and parallel execution
- Smart timeout management
- Staggered model starts to prevent resource conflicts

**Persistent Memory**
- ChromaDB-based vector storage for conversation history
- Semantic search for relevant context
- Fact tracking and inference engine
- User preference learning

**Web GUI & CLI**
- Beautiful FastAPI web interface with real-time updates
- Full-featured command-line tool
- WebSocket support for streaming responses
- Session management and chat history

---

## Quick Start

### Prerequisites

- **Python 3.8+**
- **Ollama** - [Install Ollama](https://ollama.ai/)
- **16GB+ RAM** (24GB recommended)
- **20GB+ free disk space** for models

### Installation

#### macOS / Linux

```bash
# Clone the repository
git clone https://github.com/pedropinho/ensemble-llm.git
cd ensemble-llm

# Run the automated installer
chmod +x scripts/install.sh
./scripts/install.sh
```

The installer will:
- Install Ollama (if needed)
- Create Python virtual environment
- Install all dependencies
- Download recommended models
- Run health checks

#### Windows

```powershell
# Clone or download the repository
git clone https://github.com/pedropinho/ensemble-llm.git
cd ensemble-llm

# Run the setup script
scripts\setup_windows.bat

# Start Ollama with GPU optimization
scripts\start_ollama_windows.bat
```

#### Manual Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
# Windows: Download from ollama.ai

# Pull recommended models
ollama pull llama3.2:3b
ollama pull phi3.5:latest
ollama pull qwen2.5:7b
ollama pull mistral:7b-instruct-q4_K_M
ollama pull gemma2:2b
```

---

## Usage

### Web Interface (Recommended)

Start the web GUI:

```bash
# Activate virtual environment
source venv/bin/activate

# Start web server
python run_web_gui.py
```

Then open http://localhost:8000 in your browser.

**Features:**
- Real-time chat interface
- Response streaming via WebSockets
- Session management
- Speed mode selection
- Web search toggle
- Chat history

### Command Line

```bash
# Activate virtual environment
source venv/bin/activate

# Basic query
python -m ensemble_llm.main "What is quantum computing?"

# With web search for current information
python -m ensemble_llm.main -w "What's the latest news about AI?"

# Interactive mode
python -m ensemble_llm.main -i

# Verbose mode (see all model responses)
python -m ensemble_llm.main -v "Explain Docker containers"

# Speed modes
python -m ensemble_llm.main --speed turbo "Quick question"
python -m ensemble_llm.main --speed balanced "Normal question"
python -m ensemble_llm.main --speed quality "Complex analysis needed"

# Custom models
python -m ensemble_llm.main --models llama3.2:3b phi3.5 qwen2.5:7b "Question"

# Combine options
python -m ensemble_llm.main -i -w -v --speed balanced
```

### Python API

```python
import asyncio
from ensemble_llm import EnsembleLLM

async def main():
    # Initialize ensemble
    ensemble = EnsembleLLM(
        models=['llama3.2:3b', 'phi3.5:latest', 'qwen2.5:7b'],
        enable_web_search=True,
        speed_mode='balanced'
    )

    await ensemble.async_init()

    # Query the ensemble
    response, metadata = await ensemble.ensemble_query(
        "What are the key principles of good API design?",
        verbose=True
    )

    print(f"Answer: {response}")
    print(f"Selected model: {metadata['selected_model']}")
    print(f"Consensus score: {metadata['consensus_score']:.2f}")

    await ensemble.cleanup()

asyncio.run(main())
```

---

## Architecture

### Core Components

#### 1. **Ensemble Orchestrator** (`main.py`)
- Manages multi-model query distribution
- Implements voting algorithms (consensus + quality scoring)
- Handles council mode and synthesis
- Coordinates all subsystems

#### 2. **Council & Synthesis System**
Models are aware they're collaborating:
- Each model knows its role and specialty
- Internal discussion phase with role clarity
- Winning model synthesizes all perspectives
- Automatic meta-talk filtering for clean outputs

#### 3. **Memory System** (`memory_system.py`)
- ChromaDB vector storage for semantic search
- Conversation history with embeddings
- Fact tracking and inference engine
- User preference learning

#### 4. **Learning System** (`learning_system.py`)
- Query cache with similarity matching
- Pattern recognition and optimization
- Model performance tracking
- Adaptive model selection

#### 5. **Web Search Integration** (`web_search.py`)
- Automatic detection of current-info queries
- Multi-source web scraping
- Context enhancement for model prompts

#### 6. **Performance Tracking** (`performance_tracker.py`)
- Response time monitoring
- Success rate tracking
- Model rotation based on performance
- Historical analytics

#### 7. **Web Server** (`web_server.py`)
- FastAPI with WebSocket support
- Session management
- Real-time streaming
- Static file serving

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│ User Query: "Explain quantum computing"                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
                    ┌──────────┐
                    │  Memory  │ (Check for relevant context)
                    └──────────┘
                          ↓
              ┌───────────────────────┐
              │  Web Search (if needed)│
              └───────────────────────┘
                          ↓
         ┌────────────────────────────────────────┐
         │    Council Discussion (Parallel)       │
         │                                        │
         │  ┌──────────┐  ┌──────────┐  ┌──────┐│
         │  │llama3.2  │  │  phi3.5  │  │qwen  ││
         │  │(general) │  │(reasoning)│  │(tech)││
         │  └──────────┘  └──────────┘  └──────┘│
         │       ↓              ↓           ↓    │
         │   Response      Response    Response  │
         └────────────────────────────────────────┘
                          ↓
              ┌──────────────────────┐
              │   Voting Algorithm   │
              │                      │
              │ • Similarity scoring │
              │ • Quality evaluation │
              │ • Performance history│
              └──────────────────────┘
                          ↓
                  Winner: phi3.5
                          ↓
              ┌──────────────────────┐
              │  Synthesis Phase     │
              │                      │
              │ phi3.5 combines all  │
              │ responses into one   │
              │ coherent answer      │
              └──────────────────────┘
                          ↓
              ┌──────────────────────┐
              │  Clean Response      │
              │                      │
              │ • Remove AI meta-talk│
              │ • Direct, authoritative│
              │ • No disclaimers     │
              └──────────────────────┘
                          ↓
         ┌────────────────────────────────┐
         │ Final Answer to User           │
         │ (Clean, comprehensive, balanced)│
         └────────────────────────────────┘
```

### Voting Algorithm

The system uses a weighted scoring mechanism:

**Consensus Score** (70% weight by default):
- TF-IDF vectorization of responses
- Cosine similarity matrix
- Average similarity to other responses

**Quality Score** (30% weight):
- Response length (optimal range)
- Structure (paragraphs, sentences)
- Response time (faster = bonus)
- Historical performance (if available)
- Web search usage (bonus if applicable)

Final selection: `argmax(consensus_score * 0.7 + quality_score * 0.3)`

---

## Configuration

All settings are in `ensemble_llm/config.py`:

### Council Mode

```python
COUNCIL_CONFIG = {
    "enabled": True,              # Enable council awareness
    "synthesis_mode": True,       # Winning model synthesizes all responses
    "filter_ai_meta_talk": True,  # Remove "as an AI" phrases

    # Customize prompts
    "system_prompt_template": "...",     # How models see the council
    "synthesis_prompt_template": "...",  # Synthesis instructions
}
```

### Speed Profiles

```python
SPEED_PROFILES = {
    "turbo": {
        "max_models": 2,
        "timeout": 10,
        "strategy": "race",  # First 1-2 responses
    },
    "fast": {
        "max_models": 3,
        "timeout": 15,
        "strategy": "cascade",  # Staggered starts
    },
    "balanced": {
        "max_models": 4,
        "timeout": 25,
        "strategy": "parallel",  # All models
    },
    "quality": {
        "max_models": 5,
        "timeout": 45,
        "strategy": "parallel",
    },
}
```

### Model Configurations

```python
MODEL_CONFIGS = {
    "llama3.2:3b": {
        "memory_gb": 3,
        "specialties": ["general", "conversation", "quick"],
        "timeout": 30,
    },
    # Add your models...
}
```

### Features Toggle

```python
FEATURES = {
    "web_search": True,
    "adaptive_models": True,
    "performance_tracking": True,
    "caching": True,
    "verbose_logging": True,
    # ...
}
```

---

## Model Recommendations

### By Available RAM

**16GB RAM** (3-4 small models):
- llama3.2:3b
- phi3.5:latest
- gemma2:2b

**24GB RAM** (4-5 medium models):
- llama3.2:3b
- phi3.5:latest
- qwen2.5:7b
- mistral:7b-instruct-q4_K_M
- gemma2:2b

**32GB+ RAM** (large models or 6+ small):
- mixtral:8x7b-instruct-q3_K_M
- llama3.1:70b-instruct-q2_K
- Or 6+ smaller models

### By GPU (Windows/CUDA)

See `WINDOWS_GPU_CONFIGS` in config.py for GPU-optimized recommendations.

---

## Development

### Project Structure

```
ensemble-llm/
├── ensemble_llm/          # Core package
│   ├── main.py           # Ensemble orchestrator
│   ├── config.py         # All configuration
│   ├── memory_system.py  # ChromaDB memory
│   ├── learning_system.py # Query cache & learning
│   ├── performance_tracker.py  # Model analytics
│   ├── web_server.py     # FastAPI server
│   ├── web_search.py     # Web search integration
│   ├── fast_mode.py      # Speed optimizations
│   ├── verbose_logger.py # Detailed logging
│   ├── platform_utils.py # OS-specific utils
│   └── static/           # Web GUI assets
├── scripts/              # Utility scripts
│   ├── install.sh        # Automated installer
│   ├── check_ollama.sh   # Health check
│   ├── setup_windows.bat # Windows setup
│   └── view_logs.py      # Log viewer
├── examples/             # Usage examples
├── docs/                 # Documentation
├── requirements.txt      # Python dependencies
├── run_web_gui.py       # Web server entry point
└── README.md            # This file
```

### Adding New Models

1. Add configuration to `ensemble_llm/config.py`:

```python
MODEL_CONFIGS["your-model:tag"] = {
    "memory_gb": 4,
    "specialties": ["domain", "task"],
    "timeout": 30,
    "description": "Model description",
}
```

2. Pull the model:
```bash
ollama pull your-model:tag
```

3. Use it:
```bash
python -m ensemble_llm.main --models your-model:tag llama3.2:3b "Test"
```

### Creating Custom Prompts

Edit `COUNCIL_CONFIG` in `config.py`:

```python
"system_prompt_template": """
INTERNAL SYSTEM MESSAGE:

You are {model_name}, specializing in {model_specialty}.

Council members: {council_members}

Provide your expert analysis on:
"""
```

### Extending the Web GUI

Static files are in `ensemble_llm/static/`:
- `static/index.html` - Main page
- `static/css/` - Stylesheets
- `static/js/` - JavaScript

WebSocket endpoints in `web_server.py`:
- `/ws` - Main WebSocket for queries
- `/api/` - REST endpoints

### Running Tests

```bash
# Check Ollama connection
./scripts/check_ollama.sh

# Test basic functionality
python -m ensemble_llm.main "Hello"

# View logs
python scripts/view_logs.py

# Run examples
python examples/synthesis_demo.py
```

---

## Contributing

We welcome contributions! Here's how to help:

### Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ensemble-llm.git
   cd ensemble-llm
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add docstrings to functions
   - Update documentation

4. **Test your changes**
   ```bash
   # Test CLI
   python -m ensemble_llm.main "Test query"

   # Test Web GUI
   python run_web_gui.py
   ```

5. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Describe what you changed and why
   - Reference any related issues

### Contribution Ideas

#### Bug Fixes
- Model timeout issues
- Memory leaks
- Web GUI glitches

#### Features
- New voting algorithms
- Additional web search sources
- Model specialization improvements
- UI/UX enhancements

#### Documentation
- Tutorial videos
- Use case examples
- API documentation
- Translation to other languages

#### Testing
- Unit tests for core components
- Integration tests
- Performance benchmarks

#### Design
- Web GUI improvements
- CLI output formatting
- Visualization of voting process

### Code Guidelines

**Style:**
- Use type hints where possible
- Follow PEP 8
- Keep functions focused and small
- Add comments for complex logic

**Documentation:**
- Docstrings for all public functions
- Update README for new features
- Add examples for new functionality

**Configuration:**
- All hardcoded values should go in `config.py`
- Use constants over magic numbers
- Make features toggleable via `FEATURES`

**Error Handling:**
- Always use try/except for external calls (Ollama, web, etc.)
- Log errors appropriately
- Provide fallback behavior

---

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Check available models
ollama list
```

### Memory Issues

**Reduce concurrent models:**
```bash
python -m ensemble_llm.main --models llama3.2:3b phi3.5 "Question"
```

**Use smaller models:**
```bash
ollama pull llama3.2:1b
ollama pull gemma2:2b
```

**Use turbo mode:**
```bash
python -m ensemble_llm.main --speed turbo "Question"
```

### Performance Issues

**Enable GPU acceleration** (if available):
- Ensure latest Ollama version
- Check GPU drivers
- Monitor with `nvidia-smi` (NVIDIA) or Activity Monitor (macOS)

**Optimize for your system:**
- Adjust `SPEED_PROFILES` in config.py
- Reduce `max_models` per profile
- Increase timeouts if models are slow

### Web GUI Not Loading

```bash
# Check port availability
lsof -i :8000

# Try different port
uvicorn ensemble_llm.web_server:app --port 8080
```

### No Web Search Results

- Check internet connection
- Verify `FEATURES["web_search"] = True`
- Check web_search.py for errors in logs

---

## FAQ

**Q: How many models should I run?**
A: 3-5 models is optimal. More models = better consensus but slower responses.

**Q: Can I use API models (OpenAI, Claude)?**
A: Currently only Ollama (local models) is supported. API support is planned.

**Q: Does it work offline?**
A: Yes, except for web search feature.

**Q: How much does it cost?**
A: Free! Everything runs locally. You only pay for hardware/electricity.

**Q: Can I run it on a laptop?**
A: Yes, with 16GB+ RAM and smaller models (3B parameters).

**Q: What's the difference from LangChain?**
A: LangChain is a framework for building LLM apps. Ensemble LLM is a specific application focused on multi-model consensus and synthesis.

**Q: Does it remember previous conversations?**
A: Yes, via the memory system (ChromaDB). Enable with `enable_memory=True`.

**Q: Can I customize the voting algorithm?**
A: Yes, edit `weighted_voting()` in `main.py` and adjust weights in `ENSEMBLE_CONFIG`.

---

## Performance Benchmarks

Typical response times (3 models, balanced mode):

| Query Type | Time | Notes |
|------------|------|-------|
| Simple fact | 5-8s | Cached: <0.1s |
| Explanation | 8-12s | With synthesis |
| Complex analysis | 12-18s | Quality mode recommended |
| With web search | +3-5s | Added to base time |

Model loading (first query): +2-5s per model

---

## Roadmap

### v1.1 (Current)
- Council mode with synthesis
- AI meta-talk filtering
- Web GUI
- Memory system

### v1.2 (Planned)
- [ ] API model support (OpenAI, Anthropic, etc.)
- [ ] Multi-language support
- [ ] Voting algorithm plugins
- [ ] Advanced analytics dashboard

### v1.3 (Future)
- [ ] Fine-tuning integration
- [ ] Custom model training
- [ ] Distributed execution
- [ ] Browser extension

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM runtime
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [scikit-learn](https://scikit-learn.org/) - ML algorithms for voting

---

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/pedropinho/ensemble-llm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pedropinho/ensemble-llm/discussions)
- **Documentation**: See `/docs` folder
- **Examples**: See `/examples` folder

---

**Built with for the local LLM community**
