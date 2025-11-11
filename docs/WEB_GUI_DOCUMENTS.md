# Web GUI Document Upload - Quick Start Guide

## Overview

The Ensemble LLM web GUI now includes a beautiful, user-friendly document management system that allows you to:
- Upload PDF and DOCX files with drag & drop
- View all uploaded documents in a dedicated panel
- Track upload progress in real-time
- Delete documents when no longer needed
- Ask questions about your documents automatically

## Getting Started

### 1. Launch the Web GUI

```bash
# Make sure you've installed the document processing dependencies
pip install PyPDF2 pdfplumber python-docx

# Start the web server
python run_web_gui.py

# Open your browser to
http://localhost:8000
```

### 2. Access the Document Panel

Click the purple **"Documents"** button in the top-right corner of the interface. This will open the Document Library panel from the right side.

## Uploading Documents

### Method 1: Click to Upload

1. Click the **"Documents"** button to open the panel
2. Click the purple **"Upload Document (PDF/DOCX)"** button
3. Select your PDF or DOCX file (max 50MB)
4. Watch the progress bar as your document uploads
5. See success message when complete!

### Method 2: Drag & Drop

1. Open the Documents panel
2. Drag a PDF or DOCX file from your computer
3. Drop it on the dashed border area
4. Upload starts automatically with progress feedback

## Features

### Real-Time Progress

- **Progress Bar**: Shows upload percentage (0-100%)
- **File Name**: Displays which file is being uploaded
- **Status Messages**:
  - ✅ Green = Success
  - ❌ Red = Error
  - 📤 Purple = Uploading

### Document Information

Each uploaded document shows:
- **Filename**: Original name of the file
- **Pages**: Total number of pages
- **Chunks**: How many chunks it was split into
- **File Type**: PDF or DOCX
- **Upload Time**: When it was uploaded (e.g., "2 hours ago")

### Document Actions

- **Search in Doc**: Pre-fills query input to search within that specific document
- **Delete**: Removes the document and all its chunks (requires confirmation)

### Toast Notifications

Small notifications appear in the bottom-right corner for:
- ✅ Upload success
- ❌ Upload errors
- 🗑️ Document deleted
- ℹ️ Other important messages

## Using Uploaded Documents

### Automatic Context Integration

Once documents are uploaded, they're **automatically** included in your queries! The system:

1. Searches all documents for relevant content
2. Finds the most related chunks
3. Includes them in the context sent to LLMs
4. LLMs answer using your document information

### Example Workflow

**Upload a contact list PDF:**
```
Joseph Smith
Phone: 555-1234
Address: 123 Main St, Boston, MA
```

**Then ask (anytime later):**
- "Where does Joseph live?" → "123 Main St, Boston, MA"
- "What's Joseph's phone number?" → "555-1234"
- "Who lives in Boston?" → "Joseph Smith"

The system remembers **forever** - ask next week, next month, or next year!

### Search in Specific Document

1. Click **"Search in Doc"** on any document
2. Query input gets pre-filled: `"What does filename.pdf say about "`
3. Complete your question and send
4. Get answers specifically from that document

## Error Handling

The GUI handles errors gracefully:

### File Too Large
- **Limit**: 50MB per file
- **Error**: "File too large (max 50MB)"
- **Solution**: Compress or split the PDF

### Invalid File Type
- **Supported**: PDF, DOCX
- **Error**: "Invalid file type"
- **Solution**: Convert to PDF or DOCX

### Upload Failed
- **Cause**: Network issues, server errors
- **Display**: Detailed error message
- **Solution**: Check connection, try again

### Processing Errors
- **Cause**: Corrupted file, unsupported PDF features
- **Display**: Specific error from server
- **Solution**: Try different file or format

## UI/UX Features

### Beautiful Design
- **Dark Theme**: Easy on the eyes
- **Purple Accent**: Document features use purple for consistency
- **Smooth Animations**: Slide-in panels, fade transitions
- **Responsive**: Works on different screen sizes

### Progress Feedback
- **Upload Progress**: Real-time percentage and bar
- **Loading States**: Disabled buttons during upload
- **Success Confirmation**: Green message + toast
- **Error Display**: Red message + toast with details

### Intuitive Controls
- **Drag & Drop**: Visual feedback when dragging files
- **Hover Effects**: Buttons glow and lift on hover
- **Delete Confirmation**: "Are you sure?" dialog
- **Keyboard Support**: Full keyboard navigation

### Document Count Badge
- **Purple badge** on Documents button shows number of uploads
- **Pulses** to draw attention when you have documents
- **Updates** automatically after upload/delete

## Tips & Best Practices

### For Best Results

1. **Use Clean PDFs**: Text-based PDFs work better than scanned images
2. **Descriptive Filenames**: Makes finding documents easier
3. **Organize Documents**: Keep related documents together
4. **Regular Cleanup**: Delete outdated documents to save space
5. **Check Upload Success**: Wait for green confirmation message

### Optimal File Sizes

- **Small (< 1MB)**: Uploads in seconds, ideal
- **Medium (1-10MB)**: Uploads in 5-30 seconds, good
- **Large (10-50MB)**: May take 1-2 minutes, still works

### Query Strategies

1. **Specific Questions**: "What's the revenue in Q3?" vs "Tell me about revenue"
2. **Use Keywords**: Include terms from the document
3. **Multiple Documents**: System searches all documents automatically
4. **Follow-up Questions**: Context is maintained in conversation

## Troubleshooting

### Panel Won't Open
- **Check**: Browser console for errors
- **Try**: Refresh page (Ctrl+R / Cmd+R)
- **Fix**: Clear browser cache

### Upload Stuck at 0%
- **Check**: File size and type
- **Try**: Different file
- **Fix**: Restart web server

### Documents Not Loading
- **Check**: Network tab in browser dev tools
- **Try**: Refresh panel (close and reopen)
- **Fix**: Check server logs

### Questions Not Using Documents
- **Check**: Documents uploaded successfully
- **Verify**: Document shows in panel with chunk count
- **Test**: Ask specific question with document keywords
- **Debug**: Check verbose mode logs

## Keyboard Shortcuts

- **Escape**: Close documents panel
- **Ctrl/Cmd + Enter**: Send query (in input field)

## Technical Details

### Upload Process

1. **Validation**: File type and size checked in browser
2. **Upload**: Sent to server with progress tracking (XMLHttpRequest)
3. **Processing**: Server extracts text and creates chunks
4. **Storage**: Chunks stored in ChromaDB with embeddings
5. **Confirmation**: Success message + document added to list

### Storage

- **Location**: `memory_store/` directory
- **Vector DB**: ChromaDB for semantic search
- **Metadata**: SQLite for structured data
- **Persistence**: Survives server restarts

### Search

- **Method**: Semantic similarity (cosine distance)
- **Threshold**: 0.3-0.4 relevance score
- **Results**: Top 3-5 most relevant chunks
- **Speed**: < 100ms typical

## API Integration

The Web GUI uses these backend endpoints:

```javascript
// Upload document
POST /api/documents/upload
Content-Type: multipart/form-data

// List all documents
GET /api/documents

// Delete document
DELETE /api/documents/{document_id}

// Search documents
POST /api/documents/search
Body: { "query": "search term", "n_results": 5 }
```

## Browser Compatibility

Tested and working on:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Performance

- **Concurrent Uploads**: One at a time (prevents race conditions)
- **Max File Size**: 50MB (configurable in backend)
- **Upload Speed**: Depends on connection (~1MB/s typical)
- **Search Speed**: < 100ms for typical queries
- **Memory Usage**: ~1KB per chunk stored

## Privacy & Security

- **Local Storage**: All documents stored locally on your machine
- **No Cloud**: Documents never sent to external services
- **Session-based**: Each user has isolated documents
- **Persistent**: Documents survive server restarts

## Customization

To customize limits, edit `ensemble_llm/config.py`:

```python
DOCUMENT_CONFIG = {
    "max_file_size_mb": 50,              # Change upload limit
    "chunk_size": 800,                   # Adjust chunk size
    "context_chunks_per_query": 3,       # More/less context
    "min_relevance_threshold": 0.3,      # Search sensitivity
}
```

## Support

For issues or questions:
- 📖 Read the full guide: `docs/DOCUMENT_UPLOAD.md`
- 💻 Check examples: `examples/document_upload_example.py`
- 🐛 Report bugs: GitHub Issues
- 📝 Check logs: `logs/` directory

## What's Next?

Now that you have document upload working:

1. **Try It**: Upload a PDF and ask questions
2. **Test Memory**: Upload today, ask tomorrow
3. **Multiple Docs**: Upload several documents, search across all
4. **Experiment**: Try different file types and sizes
5. **Integrate**: Use documents with web search for best results

Enjoy your enhanced Ensemble LLM experience! 🚀
