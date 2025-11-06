# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common Commands

### Setup and Installation
```bash
# Install all dependencies and models
./scripts/install.sh

# Activate virtual environment (required before running)
source venv/bin/activate

# Check Ollama health and available models
./scripts/check_ollama.sh
```

### Running the Application
```bash
# Basic query with default models
python -m ensemble_llm.main "Your question here"

# Interactive mode
python -m ensemble_llm.main -i

# Interactive with web search enabled
python -m ensemble_llm.main -i -w

# Verbose mode (shows all model responses and voting details)
python -m ensemble_llm.main -v "Your question"

# Use specific models
python -m ensemble_llm.main --models llama3.2:3b phi3.5:latest qwen2.5:7b "Your question"

# Enable web search for current information
python -m ensemble_llm.main -w "Current stock price of Apple"

# Debug mode with detailed logging
python -m ensemble_llm.main --log-level DEBUG -v "Test query"
```

### Ollama Management
```bash
# Check if Ollama is running
pgrep -x "ollama"

# Start Ollama service (if not running)
ollama serve

# List installed models
ollama list

# Pull a new model
ollama pull model_name:tag

# Remove a model
ollama rm model_name:tag

# Test a specific model directly
ollama run model_name "test prompt"
```

### Development
```bash
# Install development dependencies
pip install -r requirements.txt

# View logs
tail -f ensemble_llm.log

# No formal test suite exists yet
```

## Architecture Overview

### Core System Design

**Ensemble LLM** is a multi-model consensus system that queries multiple local LLMs in parallel and selects the best response through intelligent voting mechanisms.

#### Key Components

1. **EnsembleLLM (main.py)** - The orchestrator class that:
   - Manages multiple Ollama model connections
   - Distributes queries to all models in parallel using asyncio
   - Performs health checks before querying
   - Collects and analyzes responses
   - Implements the voting algorithm

2. **WebSearcher (web_search.py)** - Optional web search integration:
   - Uses DuckDuckGo (no API key required)
   - Automatically triggered for queries containing "current", "latest", "news", etc.
   - Can retry queries with simplified search terms
   - Enhances model prompts with current information

3. **Voting System** - Multi-factor response selection:
   - **Consensus Score (60% weight)**: TF-IDF vectorization + cosine similarity to measure agreement between models
   - **Quality Score (40% weight)**: Evaluates response length, structure, web search usage, and response time
   - Selects response with highest combined score

4. **Logger (logger.py)** - Colored console output and file logging

#### Data Flow

```
User Query
    ↓
Check Model Health (parallel health checks)
    ↓
Detect if Web Search Needed (keyword analysis)
    ↓
[Optional] Enhance Prompt with Web Search
    ↓
Query All Models in Parallel (asyncio.gather)
    ↓
Collect Responses with Retry Logic (exponential backoff)
    ↓
Detect Uncertainty in Responses
    ↓
[Optional] Retry with Web Search
    ↓
Calculate Similarity Matrix (TF-IDF + cosine similarity)
    ↓
Calculate Quality Scores (length, structure, web bonus, time)
    ↓
Weighted Voting (consensus 60% + quality 40%)
    ↓
Return Best Response + Metadata
```

### Important Implementation Details

#### Async Model Querying
- All model queries use `aiohttp` for true parallel execution
- Each query has a 60-second timeout
- Exponential backoff retry (default 2 attempts)
- Health checks run before queries to avoid wasting time on unavailable models

#### Web Search Integration
- Triggered automatically by keywords: "current", "latest", "today", "now", "recent", "2024", "2025", "news", "price", "stock", "weather"
- Also triggered if model response indicates uncertainty (phrases like "I don't have current information")
- Web search results are prepended to the original prompt
- Models get a 1.2x quality score bonus when using web search

#### Model Health Checks
- Tests model availability via `/api/tags` endpoint
- Runs a minimal test query to verify responsiveness
- Unhealthy models are excluded from the ensemble but included in response metadata

#### Scoring Algorithm
```python
# Consensus: average similarity with all other responses
consensus_score = np.mean(similarity_matrix, axis=1)

# Quality components:
# - length_score: min(len(text) / 500, 2.0) / 2.0 (prefers moderate length)
# - has_structure: 1.0 if newlines present, else 0.8
# - web_bonus: 1.2 if used web search, else 1.0
# - time_penalty: 1.0 for <5s, down to 0.8 for >20s

quality_score = length_score * has_structure * web_bonus * time_penalty

# Final score
final_score = consensus_score * 0.6 + quality_score * 0.4
```

### Configuration

#### Default Models
The system defaults to: `llama3.2:3b`, `phi3.5:latest`, `qwen2.5:7b`, `mistral:7b-instruct-q4_K_M`

#### Model Selection Guidelines
- **16GB RAM**: 3-4 small models (2-3B parameters)
- **24GB RAM** (optimal): 4-5 medium models (3-7B parameters)
- **32GB+ RAM**: 3-4 large models or 6+ small models

Refer to `docs/models.md` for detailed model recommendations and quantization levels.

#### Ollama Connection
- Default host: `http://localhost:11434`
- API endpoints used: `/api/tags`, `/api/generate`
- Models must be pre-installed via `ollama pull`

### Error Handling

The system handles several error conditions gracefully:
- **Model unavailable**: Excluded from ensemble, warning logged
- **Timeout**: Exponential backoff retry, then marked as failed
- **Connection errors**: Retry with backoff, then mark as failed
- **No successful responses**: Returns error message with metadata
- **Web search failure**: Falls back to query without web context

### Logging

- **Console**: Colored output at INFO level by default
- **File**: `ensemble_llm.log` with DEBUG level details
- Log level can be adjusted via `--log-level` CLI flag

## Project Structure

```
ensemble_llm/
├── init.py          # Package initialization, exports main classes
├── main.py          # EnsembleLLM class, CLI interface, voting logic
├── web_search.py    # WebSearcher class for DuckDuckGo integration
└── logger.py        # ColoredFormatter and setup_logger utilities

scripts/
├── install.sh       # Complete installation script (Ollama + models + Python env)
└── check_ollama.sh  # Health check script for Ollama and models

docs/
└── models.md        # Detailed model recommendations and configuration
```

## Working with This Codebase

### Adding New Models
1. Pull the model: `ollama pull new_model:tag`
2. Add to CLI defaults or specify via `--models` flag
3. No code changes required - system dynamically queries any valid Ollama model

### Modifying the Voting Algorithm
- Edit `weighted_voting()` method in `ensemble_llm/main.py`
- Adjust weights on line 392: `final_scores = (consensus_scores * 0.6 + quality_scores * 0.4)`
- Modify quality score calculation (lines 366-387) to add new factors

### Adding New Web Search Sources
- Edit `WebSearcher` class in `ensemble_llm/web_search.py`
- Add new search method similar to `search_duckduckgo()`
- Update `search_with_fallback()` to try new source

### Extending Model Capabilities
The system can be extended by modifying:
- **Prompt enhancement**: `enhance_prompt_with_web_search()` in main.py
- **Uncertainty detection**: `detect_uncertainty()` keywords list
- **Response metadata**: Add fields to the metadata dict in `query_model()`

### Python API Usage
```python
from ensemble_llm import EnsembleLLM
import asyncio

async def main():
    ensemble = EnsembleLLM(
        models=['llama3.2:3b', 'phi3.5:latest', 'qwen2.5:7b'],
        enable_web_search=True
    )
    
    response, metadata = await ensemble.ensemble_query(
        "Your question here",
        verbose=True
    )
    
    print(response)
    await ensemble.cleanup()

asyncio.run(main())
```

## Dependencies

- **aiohttp**: Async HTTP for parallel model queries
- **numpy**: Numerical operations for similarity calculations
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **beautifulsoup4** + **lxml**: Web scraping (not currently used but available)
- **colorama**: Terminal color support

## Platform Notes

- **Primary platform**: macOS (Apple Silicon optimized via Ollama)
- **Secondary platform**: Linux
- **Not supported**: Windows (Ollama support limited)
- **Ollama required**: Must be installed and running on localhost:11434

## Known Limitations

- No persistent conversation history in CLI mode
- Web search limited to DuckDuckGo (no API keys needed but limited features)
- No unit tests or integration tests currently exist
- Models must be pre-installed; no automatic download during queries
- All models must be compatible with Ollama's API format
