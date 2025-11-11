"""
Example: Document Upload and Retrieval

This example demonstrates how to:
1. Upload and process a document (PDF or DOCX)
2. Store it in the memory system
3. Search for relevant content
4. Use the document in queries with EnsembleLLM
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble_llm.document_processor import DocumentProcessor
from ensemble_llm.memory_system import SemanticMemory
from ensemble_llm.main import EnsembleLLM
from ensemble_llm.config import DEFAULT_MODELS
import asyncio


def example_1_process_and_store_document():
    """Example 1: Process a document and store it in memory"""

    print("=" * 60)
    print("Example 1: Processing and Storing a Document")
    print("=" * 60)

    # Initialize processor with custom settings
    processor = DocumentProcessor(
        chunk_size=800,      # ~800 tokens per chunk
        chunk_overlap=200,   # 200 tokens overlap for context
        min_chunk_size=100   # Minimum chunk size
    )

    # Check what file types are supported
    print(f"\nPDF support: {processor.pdf_support}")
    print(f"DOCX support: {processor.docx_support}")
    print(f"DOC support: {processor.doc_support}")

    # For this example, we'll simulate a document
    # In real usage, replace with your actual document path
    doc_path = Path("path/to/your/document.pdf")

    if not doc_path.exists():
        print(f"\n⚠️  Document not found: {doc_path}")
        print("Please replace 'path/to/your/document.pdf' with an actual file path")
        print("\nSkipping document processing in this example...")
        return

    # Process the document
    print(f"\nProcessing document: {doc_path.name}")
    processed_doc = processor.process_document(doc_path)

    print(f"\n✓ Document processed successfully!")
    print(f"  - Filename: {processed_doc.filename}")
    print(f"  - File type: {processed_doc.file_type}")
    print(f"  - Total pages: {processed_doc.total_pages}")
    print(f"  - Total chunks: {processed_doc.total_chunks}")
    print(f"  - Document ID: {processed_doc.document_id}")

    # Show first chunk preview
    if processed_doc.chunks:
        first_chunk = processed_doc.chunks[0]
        preview = first_chunk.content[:200] + "..." if len(first_chunk.content) > 200 else first_chunk.content
        print(f"\n📄 First chunk preview:")
        print(f"  {preview}")

    # Initialize memory system
    memory = SemanticMemory(memory_dir="memory_store")

    # Store document in memory
    print(f"\nStoring document in memory system...")
    chunks_data = [chunk.to_dict() for chunk in processed_doc.chunks]

    document_id = memory.store_document(
        document_id=processed_doc.document_id,
        filename=processed_doc.filename,
        file_type=processed_doc.file_type,
        total_pages=processed_doc.total_pages,
        total_chunks=processed_doc.total_chunks,
        chunks=chunks_data,
        metadata={
            "processed_at": processed_doc.processed_at.isoformat(),
            "example": "document_upload_example.py"
        }
    )

    print(f"✓ Document stored with ID: {document_id}")

    return document_id


def example_2_search_documents():
    """Example 2: Search for content in uploaded documents"""

    print("\n" + "=" * 60)
    print("Example 2: Searching Documents")
    print("=" * 60)

    # Initialize memory
    memory = SemanticMemory(memory_dir="memory_store")

    # Get all uploaded documents
    all_docs = memory.get_all_documents()

    print(f"\n📚 Total documents in memory: {len(all_docs)}")

    if not all_docs:
        print("No documents found. Upload a document first using Example 1.")
        return

    # Show all documents
    print("\nUploaded documents:")
    for idx, doc in enumerate(all_docs, 1):
        print(f"\n  {idx}. {doc['filename']}")
        print(f"     - ID: {doc['document_id']}")
        print(f"     - Type: {doc['file_type']}")
        print(f"     - Pages: {doc['total_pages']}, Chunks: {doc['total_chunks']}")
        print(f"     - Uploaded: {doc['upload_date']}")

    # Search for specific content
    search_query = "machine learning"
    print(f"\n🔍 Searching for: '{search_query}'")

    results = memory.search_documents(
        query=search_query,
        n_results=5,
        min_relevance=0.3
    )

    print(f"\n✓ Found {len(results)} relevant chunks:")

    for idx, result in enumerate(results, 1):
        filename = result['metadata'].get('filename', 'Unknown')
        chunk_idx = result['metadata'].get('chunk_index', 0)
        relevance = result['relevance']

        # Show snippet
        content_preview = result['content'][:150]
        if len(result['content']) > 150:
            content_preview += "..."

        print(f"\n  {idx}. [{filename}, chunk {chunk_idx + 1}] (relevance: {relevance:.2f})")
        print(f"     {content_preview}")


def example_3_query_with_document_context():
    """Example 3: Query LLMs with automatic document context"""

    print("\n" + "=" * 60)
    print("Example 3: Querying with Document Context")
    print("=" * 60)

    async def run_query():
        # Initialize Ensemble LLM
        ensemble = EnsembleLLM(
            models=DEFAULT_MODELS[:3],  # Use first 3 models for speed
            verbose=True  # Show which document chunks are used
        )

        # Example query that would benefit from document context
        query = "What are the main findings from the research?"

        print(f"\n📝 Query: {query}")
        print("\nProcessing with document context...\n")

        # The memory system will automatically include relevant document chunks
        response, metadata = await ensemble.ensemble_query(query)

        print(f"\n✅ Response from {metadata.get('selected_model', 'ensemble')}:")
        print(f"\n{response}\n")

        # Show voting details if available
        if 'all_scores' in metadata:
            print("\n📊 Voting scores:")
            for model, score in metadata['all_scores'].items():
                print(f"  - {model}: {score:.2f}")

        # Cleanup
        await ensemble.cleanup()

    # Run the async query
    asyncio.run(run_query())


def example_4_delete_document():
    """Example 4: Delete a document from memory"""

    print("\n" + "=" * 60)
    print("Example 4: Deleting Documents")
    print("=" * 60)

    # Initialize memory
    memory = SemanticMemory(memory_dir="memory_store")

    # Get all documents
    all_docs = memory.get_all_documents()

    if not all_docs:
        print("No documents to delete.")
        return

    print("\nAvailable documents:")
    for idx, doc in enumerate(all_docs, 1):
        print(f"  {idx}. {doc['filename']} (ID: {doc['document_id']})")

    # In a real application, you would ask the user which to delete
    # For this example, we'll just show how to delete
    print("\n⚠️  To delete a document, use:")
    print("  memory.delete_document(document_id)")
    print("\nExample (commented out to prevent accidental deletion):")
    print("  # document_id = all_docs[0]['document_id']")
    print("  # success = memory.delete_document(document_id)")
    print("  # if success:")
    print("  #     print(f'✓ Deleted document {document_id}')")


def main():
    """Run all examples"""

    print("\n" + "="*60)
    print("Document Upload and Retrieval Examples")
    print("="*60)

    print("""
This script demonstrates the document upload feature:

1. Process and store a document
2. Search for content in documents
3. Query with automatic document context
4. Delete documents

Note: Examples 1 and 3 require actual document files.
      Replace the file paths with your own documents to test.
    """)

    # Run examples
    try:
        # Example 1: Process and store (requires actual file)
        doc_id = example_1_process_and_store_document()

        # Example 2: Search documents
        example_2_search_documents()

        # Example 3: Query with context (requires Ollama running)
        # Uncomment to test with actual LLMs:
        # example_3_query_with_document_context()

        # Example 4: Delete document
        example_4_delete_document()

        print("\n" + "="*60)
        print("Examples completed!")
        print("="*60)

        print("""
Next steps:
1. Replace 'path/to/your/document.pdf' with an actual PDF or DOCX file
2. Run the examples again to see document processing in action
3. Try different search queries to test semantic search
4. Uncomment Example 3 to test with live LLMs
5. Check the docs/DOCUMENT_UPLOAD.md for complete guide
        """)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
