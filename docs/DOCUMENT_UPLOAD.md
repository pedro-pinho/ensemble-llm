# Document Upload and Retrieval Guide

This guide explains how to use the document upload feature in Ensemble LLM to upload PDF and DOCX files, and ask questions about their content with long-term memory retention.

## Features

- **Supported Formats**: PDF (.pdf) and Microsoft Word (.docx)
- **Intelligent Chunking**: Documents are split into overlapping chunks for better context
- **Semantic Search**: Uses vector embeddings to find relevant content
- **Long-term Memory**: Document content is stored persistently in ChromaDB
- **Automatic Context**: Relevant document chunks are automatically included in queries

## Installation

### 1. Install Required Dependencies

The document processing feature requires additional Python libraries:

```bash
# Activate your virtual environment first
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install document processing dependencies
pip install PyPDF2 pdfplumber python-docx

# Or reinstall all requirements
pip install -r requirements.txt
```

### 2. Verify Installation

Run the following to check if document processing is available:

```bash
python -c "from ensemble_llm.document_processor import DocumentProcessor; print('Document processing ready!')"
```

## Usage

### Web GUI (Recommended)

1. **Start the Web Server**:
   ```bash
   python run_web_gui.py
   # or
   ./scripts/start_web_gui.sh  # macOS/Linux
   scripts\start_web_gui.bat   # Windows
   ```

2. **Open Browser**: Navigate to `http://localhost:8000`

3. **Upload Document**:
   - Click the "Upload Document" button
   - Select a PDF or DOCX file (max 50MB)
   - Wait for processing confirmation

4. **Ask Questions**: Once uploaded, simply ask questions:
   - "What is this document about?"
   - "Summarize the key points"
   - "What does it say about [topic]?"

5. **Manage Documents**:
   - View all uploaded documents in the Documents panel
   - Delete documents you no longer need
   - Search specific content within documents

### Programmatic Usage

```python
from pathlib import Path
from ensemble_llm.document_processor import DocumentProcessor
from ensemble_llm.memory_system import SemanticMemory

# Initialize
processor = DocumentProcessor(
    chunk_size=800,
    chunk_overlap=200,
)

memory = SemanticMemory(memory_dir="memory_store")

# Process a document
doc_path = Path("path/to/your/document.pdf")
processed_doc = processor.process_document(doc_path)

# Store in memory
chunks_data = [chunk.to_dict() for chunk in processed_doc.chunks]
memory.store_document(
    document_id=processed_doc.document_id,
    filename=processed_doc.filename,
    file_type=processed_doc.file_type,
    total_pages=processed_doc.total_pages,
    total_chunks=processed_doc.total_chunks,
    chunks=chunks_data,
)

# Search documents
results = memory.search_documents(
    query="What are the main findings?",
    n_results=5,
    min_relevance=0.3,
)

for result in results:
    print(f"Relevance: {result['relevance']:.2f}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Source: {result['metadata']['filename']}")
    print()
```

## How It Works

### 1. Document Processing

When you upload a document:

1. **Text Extraction**:
   - PDF: Uses `pdfplumber` (preferred) or `PyPDF2` as fallback
   - DOCX: Uses `python-docx` to extract text from paragraphs and tables

2. **Cleaning**:
   - Removes excessive whitespace
   - Normalizes line endings
   - Filters control characters

3. **Chunking**:
   - Splits text into ~800 token chunks (configurable)
   - Creates 200 token overlap between chunks for context continuity
   - Attempts to break at sentence boundaries
   - Each chunk tracks its position, source file, and page number

### 2. Storage

- **Vector Storage (ChromaDB)**: Each chunk is embedded using `sentence-transformers` and stored for semantic search
- **Structured Storage (SQLite)**: Metadata like filename, upload date, chunk index stored for fast lookups
- **Metadata**: Every chunk includes:
  - `document_id`: Unique identifier
  - `filename`: Original filename
  - `chunk_index`: Position in document (0-based)
  - `total_chunks`: Total number of chunks
  - `upload_date`: When it was uploaded

### 3. Retrieval

When you ask a question:

1. **Semantic Search**: Your query is compared to all document chunks using cosine similarity
2. **Relevance Filtering**: Only chunks above the relevance threshold (default 0.3) are returned
3. **Context Integration**: Top 3 relevant chunks are automatically included in the context sent to LLMs
4. **Model Processing**: LLMs receive both your question and relevant document excerpts

### 4. Long-term Memory

- Documents persist in `memory_store/` directory
- Survives restarts and sessions
- Can be queried months later
- Example: Upload a contact list today, ask "Where does Joseph live?" next month

## Configuration

Edit `ensemble_llm/config.py` to customize:

```python
DOCUMENT_CONFIG = {
    "chunk_size": 800,              # Tokens per chunk
    "chunk_overlap": 200,           # Overlap between chunks
    "min_chunk_size": 100,          # Minimum chunk size
    "max_file_size_mb": 50,         # Max upload size
    "allowed_extensions": [".pdf", ".docx"],
    "default_search_results": 5,    # Chunks to retrieve
    "min_relevance_threshold": 0.3, # Minimum similarity
    "context_chunks_per_query": 3,  # Chunks in context
    "auto_include_in_context": True,
}
```

## API Endpoints

### Upload Document
```
POST /api/documents/upload
Content-Type: multipart/form-data

Response:
{
  "status": "success",
  "document_id": "doc_abc123...",
  "filename": "report.pdf",
  "total_pages": 42,
  "total_chunks": 85
}
```

### List Documents
```
GET /api/documents

Response:
{
  "status": "success",
  "documents": [
    {
      "document_id": "doc_abc123...",
      "filename": "report.pdf",
      "file_type": ".pdf",
      "total_pages": 42,
      "total_chunks": 85,
      "upload_date": "2025-01-15T10:30:00"
    }
  ]
}
```

### Delete Document
```
DELETE /api/documents/{document_id}

Response:
{
  "status": "success",
  "message": "Document deleted successfully"
}
```

### Search Documents
```
POST /api/documents/search
Content-Type: application/json

{
  "query": "machine learning algorithms",
  "n_results": 5,
  "min_relevance": 0.3
}

Response:
{
  "status": "success",
  "results": [
    {
      "content": "...",
      "metadata": {...},
      "relevance": 0.87
    }
  ]
}
```

## Best Practices

### Document Preparation

1. **Clean PDFs**: Use PDFs with selectable text (not scanned images)
2. **Optimize Size**: Keep documents under 50MB for faster processing
3. **Clear Formatting**: Well-formatted documents extract better
4. **Use DOCX for Office Docs**: Convert .doc files to .docx for better support

### Query Optimization

1. **Be Specific**: "What are the revenue figures in Q3?" vs "Tell me about revenue"
2. **Use Keywords**: Include terms likely to appear in the document
3. **Multi-step Queries**: For complex questions, break into smaller queries
4. **Context Matters**: The system shows you which document chunks were used

### Memory Management

1. **Regular Cleanup**: Delete outdated documents to save space
2. **Descriptive Filenames**: Use clear names for easier reference
3. **Organize by Topic**: Upload related documents together
4. **Monitor Storage**: Check `memory_store/` directory size periodically

## Troubleshooting

### "No PDF processing library found"

```bash
pip install PyPDF2 pdfplumber
```

### "python-docx not found"

```bash
pip install python-docx
```

### "Failed to extract PDF"

- Ensure PDF contains selectable text (not scanned images)
- Try using a different PDF reader to re-save the file
- Use OCR tools for scanned documents first

### "Document too large"

- Compress the PDF using online tools
- Split large documents into smaller parts
- Increase `max_file_size_mb` in config (may slow processing)

### Poor Search Results

- Try different keywords
- Reduce `min_relevance_threshold` in search
- Check if document uploaded successfully
- Verify text extracted correctly (check logs)

## Examples

### Example 1: Resume Analysis

```python
# Upload a resume
# Then ask:
"What programming languages does this person know?"
"Summarize their work experience"
"Do they have experience with machine learning?"
```

### Example 2: Research Paper

```python
# Upload a research paper
# Then ask:
"What is the main hypothesis?"
"What methodology did they use?"
"What were the key findings?"
"How does this compare to previous research?"
```

### Example 3: Contact Directory

```python
# Upload a contact list PDF
# Store: Joseph Smith - 555-1234 - 123 Main St, Boston
# Later (even months later):
"Where does Joseph live?"
"What's Joseph's phone number?"
```

### Example 4: Meeting Notes

```python
# Upload meeting notes DOCX
# Then ask:
"What action items were assigned?"
"When is the next deadline?"
"What did Sarah say about the budget?"
```

## Architecture Diagram

```
┌─────────────┐
│   User      │
│  (Web GUI)  │
└──────┬──────┘
       │ Upload PDF/DOCX
       ▼
┌─────────────────────────────┐
│  DocumentProcessor          │
│  - Extract text             │
│  - Clean & normalize        │
│  - Create overlapping chunks│
└──────────┬──────────────────┘
           │ Chunks
           ▼
┌─────────────────────────────┐
│  SemanticMemory             │
│  ┌──────────┬──────────┐   │
│  │ ChromaDB │  SQLite  │   │
│  │ (vectors)│(metadata)│   │
│  └──────────┴──────────┘   │
└──────────┬──────────────────┘
           │
    ┌──────┴───────┐
    │              │
    ▼              ▼
Query → Search → Context → LLM → Answer
```

## Performance Notes

- **Upload Time**: ~2-10 seconds for typical documents
- **Chunk Size**: 800 tokens ≈ 3200 characters ≈ ~1 page
- **Memory Usage**: ~1KB per chunk in vector DB
- **Search Speed**: <100ms for typical queries
- **Concurrent Uploads**: Supported (separate sessions)

## Future Enhancements

Potential improvements being considered:

- [ ] Image extraction from PDFs
- [ ] Support for more formats (TXT, MD, HTML)
- [ ] OCR for scanned documents
- [ ] Document summarization on upload
- [ ] Multi-document comparison queries
- [ ] Export document knowledge base
- [ ] Document versioning
- [ ] Collaborative document annotations

## Support

For issues or questions:
- Check logs in `logs/` directory
- Review memory database in `memory_store/`
- See main README for general troubleshooting
- Report bugs at [GitHub Issues](https://github.com/anthropics/ensemble-llm/issues)
