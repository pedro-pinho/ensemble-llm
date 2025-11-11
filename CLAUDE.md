# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ensemble LLM is a multi-model local LLM system that runs multiple language models in parallel via Ollama and uses intelligent voting mechanisms to select the best answer. It features web search integration, memory systems, adaptive learning, and performance tracking.

## Core Architecture

### Main Components

1. **EnsembleLLM** (`ensemble_llm/main.py`): Core orchestrator that manages query distribution, response collection, and consensus voting
   - Distributes queries to multiple models in parallel using `asyncio`
   - Collects responses and calculates consensus using TF-IDF vectorization and cosine similarity
   - Integrates all other subsystems (memory, learning, performance tracking, web search)

2. **Memory System** (`ensemble_llm/memory_system.py`): ChromaDB-based persistent memory with semantic search
   - Uses `sentence-transformers` for embeddings and ChromaDB for vector storage
   - Stores facts, preferences, conversations, and inferences
   - Includes `InferenceEngine` for deriving new knowledge from stored facts
   - Metadata must be sanitized using `sanitize_metadata()` before ChromaDB storage (only supports primitives)

3. **Learning System** (`ensemble_llm/learning_system.py`): Query caching, pattern recognition, and continuous optimization
   - `QueryCache`: Similarity-based query caching to avoid redundant LLM calls
   - `SmartEnsembleOrchestrator`: Analyzes query patterns to select optimal models
   - `PrecomputeManager`: Precomputes common queries in background
   - `CacheManager`: Manages cache lifecycle and statistics

4. **Document Processing** (`ensemble_llm/document_processor.py`): PDF and DOCX document upload and processing
   - `DocumentProcessor`: Extracts text from PDF/DOCX files and splits into chunks
   - Intelligent chunking with configurable overlap for context continuity
   - Integrates with memory system for long-term document storage
   - Supports semantic search across uploaded documents

5. **Performance Tracking** (`ensemble_llm/performance_tracker.py`): Adaptive model selection and rotation
   - Tracks success rates, response times, selection rates per model
   - `AdaptiveModelManager`: Dynamically swaps underperforming models
   - Uses rolling windows for recent performance evaluation
   - Logs detailed metrics when verbose logging enabled

6. **Fast Mode** (`ensemble_llm/fast_mode.py`): Speed optimization strategies
   - Race strategy: Returns first N responses
   - Cascade strategy: Starts fast models first, adds more if needed
   - Single best: Uses only highest-performing model
   - `TurboMode` and `ModelWarmup` for performance optimization

7. **Web Server** (`ensemble_llm/web_server.py`): FastAPI + WebSocket interface for web GUI
   - Session management with chat history
   - Real-time query processing via WebSockets
   - Document upload endpoints (/api/documents/upload, /api/documents, etc.)
   - Serves static files from `ensemble_llm/static/`

8. **Configuration** (`ensemble_llm/config.py`): Centralized configuration with platform-specific settings
   - Platform detection (macOS/Windows/Linux) with optimized defaults
   - Model pools organized by specialty (code, math, creative)
   - Speed profiles (turbo/fast/balanced/quality)
   - Windows GPU optimization configs by VRAM size

## Key Technical Details

### Consensus Algorithm
- Uses scikit-learn's TfidfVectorizer to convert responses to vectors
- Calculates pairwise cosine similarity between all model responses
- Consensus score = average similarity to all other responses
- Quality score based on response length, structure, and detail
- Final score = weighted combination (configurable via `ENSEMBLE_CONFIG`)

### Asynchronous Architecture
- All model queries use `asyncio` and `aiohttp` for parallelism
- Critical: Always use `async`/`await` when calling model endpoints
- Session management uses single `aiohttp.ClientSession` per instance
- Must call `cleanup()` to properly close sessions

### Memory System Integration
- Memory is queried before sending to models to add context
- Uses semantic search to find relevant past interactions
- All metadata stored in ChromaDB must be primitives (use `sanitize_metadata()`)
- Memory storage path: `memory_store/` directory
- Automatically includes relevant document chunks in query context
- Documents stored with `MemoryType.DOCUMENT` for semantic retrieval

### Platform-Specific Behavior
- Windows: Uses high process priority, GPU layer optimization, VRAM-based model selection
- macOS: Optimized for Apple Silicon with Metal acceleration
- Linux: Standard configuration with CUDA support
- Platform utilities in `platform_utils.py` detect and optimize for each OS

## Common Development Commands

### Running the Application

```bash
# Activate virtual environment first
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# CLI mode - basic query
python -m ensemble_llm.main "Your question here"

# With web search
python -m ensemble_llm.main -w "What's the latest news about AI?"

# Interactive mode
python -m ensemble_llm.main -i

# Verbose mode (see all model responses)
python -m ensemble_llm.main -v "Question"

# Fast mode (speed profiles: turbo/fast/balanced/quality)
python -m ensemble_llm.main --speed fast "Question"

# Custom models
python -m ensemble_llm.main --models llama3.2:3b phi3.5 "Question"

# Web GUI (includes document upload interface)
python run_web_gui.py
# Then open http://localhost:8000

# Upload documents via web GUI or API
# See docs/DOCUMENT_UPLOAD.md for detailed guide
```

### Setup and Installation

```bash
# Automated installation (macOS/Linux)
chmod +x scripts/install.sh
./scripts/install.sh

# Windows setup
scripts\setup_windows.bat

# Manual installation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Document processing dependencies (for PDF/DOCX upload)
pip install PyPDF2 pdfplumber python-docx

# Pull Ollama models
ollama pull llama3.2:3b
ollama pull phi3.5:latest
ollama pull qwen2.5:7b
ollama pull mistral:7b-instruct-q4_K_M
ollama pull gemma2:2b
```

### Utility Scripts

```bash
# Check Ollama setup and available models
./scripts/check_ollama.sh

# View detailed logs with filtering
python scripts/view_logs.py

# Fix memory database issues
python scripts/fix_memory_db.py

# Windows GPU benchmark
python scripts/benchmark_windows.py

# Start web GUI
./scripts/start_web_gui.sh  # macOS/Linux
scripts\start_web_gui.bat   # Windows
```

### Testing

There are no unit tests in this repository yet. When adding tests:
- Create `tests/` directory in project root
- Use `pytest` as the testing framework (add to requirements.txt)
- Test async code using `pytest-asyncio`

## Important Constraints and Patterns

### When Modifying the Ensemble Logic
- The voting mechanism is in `EnsembleLLM.ensemble_query()` in main.py
- Consensus weights are configured in `ENSEMBLE_CONFIG` in config.py
- Must handle empty responses gracefully (models can fail/timeout)
- Use `ERROR_MESSAGES` and `SUCCESS_MESSAGES` from config.py for user-facing messages

### When Adding New Models
- Add model configuration to `MODEL_CONFIGS` in config.py
- Include memory requirements, specialties, timeout values
- Update appropriate model pool (`MODEL_POOLS`)
- Test on target platform to verify memory requirements

### When Working with Memory System
- Always sanitize metadata with `sanitize_metadata()` before ChromaDB operations
- ChromaDB only accepts: str, int, float, bool (no nested dicts/lists)
- Use `MemoryType` enum for memory categorization
- Memory queries return similarity scores - use threshold filtering

### When Modifying Web Server
- Static files go in `ensemble_llm/static/` (css/, js/ subdirectories)
- WebSocket communication uses JSON messages
- Session cleanup runs periodically - see `cleanup_old_sessions()`
- Each session gets its own `EnsembleLLM` instance

### Configuration Management
- Never hardcode values - use config.py constants
- Platform-specific settings use `IS_WINDOWS`, `IS_MACOS`, `IS_LINUX` flags
- Speed profiles defined in `SPEED_PROFILES` - map to model sets and strategies
- Logging levels controlled via `LOGGING_CONFIG`
- Document processing settings in `DOCUMENT_CONFIG` (chunk size, file limits, etc.)

### Council Mode (Model Awareness)
- Council mode makes models aware they're part of an ensemble
- Configured via `COUNCIL_CONFIG` in config.py
- When enabled, each model receives context about:
  - Being part of a council of AI models
  - Their specific specialty/role
  - Names of other council members
  - That final answer is selected by consensus
- Prompt modification happens in `create_council_aware_prompt()` (main.py:384)
- Applied to both fast and optimized query paths
- Can be extended for iterative/debate modes (see examples/iterative_council.py)

### Synthesis Mode (Answer Refinement)
- When `synthesis_mode: True` in COUNCIL_CONFIG, adds a final synthesis step
- Flow: Models discuss → Voting selects winner → Winner synthesizes all responses
- The winning model acts as "spokesperson" to combine all insights
- Synthesis happens in `synthesize_final_answer()` (main.py:1068)
- Synthesis prompt explicitly instructs model to:
  - Combine best insights from all council members
  - Resolve contradictions between perspectives
  - Remove meta-discussion (no mention of council/voting)
  - Provide unified, coherent answer to user
- Benefits: More comprehensive answers, cleaner output, combines diverse perspectives
- Fallback: If synthesis fails, returns original winning response (also filtered)
- Note: Synthesis skipped in turbo mode for speed

### AI Meta-Talk Filtering
- When `filter_ai_meta_talk: True`, post-processes responses to remove AI self-references
- Filter implemented in `filter_ai_meta_talk()` (main.py:411)
- Removes entire sentences containing patterns like:
  - "as an AI", "I don't have access to", "as a language model"
  - "the council discussed", "based on my training", "I cannot"
  - See `meta_talk_patterns` in COUNCIL_CONFIG for full list
- Applied to all synthesis outputs and fallback responses
- Uses regex with sentence-level removal (removes whole sentence, not just phrase)
- Makes final answers more direct and authoritative
- Configurable: add custom patterns to `meta_talk_patterns` array

### Role Clarity Improvements
- Council prompts now explicitly distinguish:
  - "YOU are an AI model" (addressing the model)
  - "The USER is a human" (addressing the question asker)
  - "INTERNAL SYSTEM MESSAGE" header for council context
  - "The user does NOT see this council process"
- Prevents confusion where models think user is part of council
- Synthesis prompt emphasizes creating response "for the USER" (human)
- Results in clearer understanding of internal discussion vs external answer

### Error Handling
- Model query failures are logged but don't stop other models
- Use try/except around all Ollama API calls
- Network errors should use exponential backoff (see retry logic in main.py)
- Always provide fallback behavior if all models fail

## Data Directories

- `cache/` - Query cache and precomputed results
- `data/` - Performance tracking data
- `docs/` - Documentation including DOCUMENT_UPLOAD.md
- `logs/` - Application logs with timestamps
- `memory_store/` - ChromaDB persistent memory storage (includes uploaded documents)
- `smart_data/` - Learning system data (patterns, statistics)
- `specialization_data/` - Model specialization metrics

## Dependencies

Core dependencies (see requirements.txt):
- `aiohttp` - Async HTTP for Ollama API calls
- `scikit-learn` - TF-IDF vectorization and similarity calculations
- `chromadb` - Vector database for memory system
- `sentence-transformers` - Text embeddings for memory
- `fastapi` + `uvicorn` - Web server
- `beautifulsoup4` - Web search result parsing
- `numpy` - Numerical operations for scoring
- `sqlalchemy` - Database operations for memory system
- `psutil` - System resource monitoring
- `PyPDF2` + `pdfplumber` - PDF document processing
- `python-docx` - DOCX document processing

## Document Upload Feature

### Overview
The document upload feature allows users to upload PDF and DOCX files, which are:
1. Extracted and cleaned
2. Split into overlapping chunks for better context
3. Stored in ChromaDB with semantic embeddings
4. Automatically retrieved when relevant to user queries
5. Persisted for long-term memory (months/years)

### Key Files
- `ensemble_llm/document_processor.py` - Document extraction and chunking
- `ensemble_llm/memory_system.py` - Document storage methods (store_document, search_documents, etc.)
- `ensemble_llm/web_server.py` - Upload endpoints
- `ensemble_llm/config.py` - DOCUMENT_CONFIG settings
- `docs/DOCUMENT_UPLOAD.md` - Complete user guide

### Usage Example
```python
# Via API
POST /api/documents/upload
# Upload PDF/DOCX file

# Via Memory System
from ensemble_llm.document_processor import DocumentProcessor
from ensemble_llm.memory_system import SemanticMemory

processor = DocumentProcessor(chunk_size=800, chunk_overlap=200)
memory = SemanticMemory()

# Process document
processed = processor.process_document(Path("report.pdf"))

# Store chunks
memory.store_document(
    document_id=processed.document_id,
    filename=processed.filename,
    file_type=processed.file_type,
    total_pages=processed.total_pages,
    total_chunks=processed.total_chunks,
    chunks=[chunk.to_dict() for chunk in processed.chunks]
)

# Later, queries automatically retrieve relevant chunks
results = memory.search_documents("machine learning", n_results=5)
```

### Integration with Main Query Pipeline
- Document chunks are automatically included in `get_user_context()`
- When a user asks a question, the memory system:
  1. Searches for relevant document chunks (semantic similarity)
  2. Filters by relevance threshold (default 0.4)
  3. Includes top 3 chunks in the context sent to LLMs
  4. LLMs receive both the question and relevant document excerpts

### Important Notes
- Documents are chunked at ~800 tokens with 200 token overlap
- Chunk overlap ensures context continuity across splits
- All metadata must be sanitized before ChromaDB storage
- File uploads limited to 50MB by default (configurable)
- Supported formats: PDF (via PyPDF2/pdfplumber), DOCX (via python-docx)

For complete documentation, see `docs/DOCUMENT_UPLOAD.md`

## Known Issues

- ChromaDB may show urllib3 warnings (suppressed in memory_system.py)
- Windows requires Ollama to be manually started before use
- Large models (13B+) may exceed memory on systems with <32GB RAM
- Web search depends on external search APIs (may have rate limits)
- PDF extraction quality depends on source (selectable text vs scanned images)
