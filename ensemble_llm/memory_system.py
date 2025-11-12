"""Intelligent Memory System for Ensemble LLM"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import sqlite3
from sentence_transformers import SentenceTransformer

from .config import MEMORY_CONFIG

warnings.filterwarnings("ignore", category=UserWarning, message=".*urllib3 v2.*")

logger = logging.getLogger("EnsembleLLM.Memory")


class MemoryType(Enum):
    """Types of memory entries"""

    FACT = "fact"
    PREFERENCE = "preference"
    CONVERSATION = "conversation"
    INFERENCE = "inference"
    RELATIONSHIP = "relationship"
    DOCUMENT = "document"


@dataclass
class MemoryEntry:
    """A single memory entry"""

    id: str
    type: MemoryType
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    confidence: float = 1.0
    source: str = "user"

    def to_dict(self):
        data = asdict(self)
        data["type"] = self.type.value
        data["timestamp"] = self.timestamp.isoformat()
        return data


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize metadata for ChromaDB storage.
    Converts complex nested structures to JSON strings.
    """
    if not metadata:
        return {}

    sanitized = {}

    for key, value in metadata.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            # These types are fine
            sanitized[key] = value
        elif isinstance(value, dict):
            # Serialize nested dicts as JSON strings
            sanitized[f"{key}_json"] = json.dumps(value)
        elif isinstance(value, list):
            # Serialize lists as JSON strings
            sanitized[f"{key}_json"] = json.dumps(value)
        elif isinstance(value, datetime):
            # Convert datetime to ISO string
            sanitized[key] = value.isoformat()
        else:
            # Convert everything else to string
            sanitized[key] = str(value)

    return sanitized


class InferenceEngine:
    """Make intelligent inferences from facts"""

    def __init__(self):
        # Define inference rules
        self.rules = [
            {
                "pattern": {"location_country": "Brazil"},
                "inferences": [
                    {"type": "language", "value": "Portuguese", "confidence": 0.9},
                    {"type": "timezone", "value": "BRT/BRST", "confidence": 0.8},
                ],
            },
            {
                "pattern": {"location_city": "São Paulo"},
                "inferences": [
                    {"type": "location_country", "value": "Brazil", "confidence": 0.95},
                    {"type": "location_state", "value": "SP", "confidence": 0.95},
                ],
            },
            {
                "pattern": {"location_city": "Bauru"},
                "inferences": [
                    {"type": "location_country", "value": "Brazil", "confidence": 0.95},
                    {"type": "location_state", "value": "SP", "confidence": 0.95},
                ],
            },
            {
                "pattern": {"profession": "software developer"},
                "inferences": [
                    {"type": "skill", "value": "programming", "confidence": 0.95},
                    {"type": "skill", "value": "problem solving", "confidence": 0.9},
                ],
            },
        ]

    def infer(self, facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate inferences from known facts"""

        inferences = []

        for rule in self.rules:
            pattern_match = all(
                facts.get(key) == value for key, value in rule["pattern"].items()
            )

            if pattern_match:
                for inference in rule["inferences"]:
                    if facts.get(inference["type"]) != inference["value"]:
                        inferences.append(
                            {
                                "type": inference["type"],
                                "value": inference["value"],
                                "confidence": inference["confidence"],
                                "source": "inference",
                                "based_on": rule["pattern"],
                            }
                        )

        return inferences


class SemanticMemory:
    """Semantic memory using ChromaDB for vector search"""

    def __init__(self, memory_dir: str = "memory_store"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

        # Add to .gitignore
        gitignore_path = self.memory_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text("*\n!.gitignore\n")

        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.memory_dir / "chromadb"),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Initialize sentence transformer for embeddings
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )

        # Create collections
        self.facts_collection = self.chroma_client.get_or_create_collection(
            name="facts",
            embedding_function=self.embedding_function,
            metadata={"description": "User facts and information"},
        )

        self.conversations_collection = self.chroma_client.get_or_create_collection(
            name="conversations",
            embedding_function=self.embedding_function,
            metadata={"description": "Conversation history"},
        )

        self.documents_collection = self.chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function,
            metadata={"description": "Uploaded document chunks"},
        )

        # Initialize SQLite for structured data
        self.db_path = self.memory_dir / "memory.db"
        self.init_database()

        # Initialize inference engine
        self.inference_engine = InferenceEngine()

        logger.info(f"Initialized semantic memory at {self.memory_dir}")

    def init_database(self):
        """Initialize SQLite database for structured data"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Facts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                category TEXT,
                key TEXT,
                value TEXT,
                confidence REAL,
                source TEXT,
                timestamp DATETIME,
                metadata TEXT,
                UNIQUE(category, key)
            )
        """
        )

        # Conversations table - CREATE THIS HERE!
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                query TEXT,
                response TEXT,
                metadata TEXT,
                timestamp DATETIME
            )
        """
        )

        # Conversation summaries table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id TEXT PRIMARY KEY,
                date DATE,
                summary TEXT,
                key_topics TEXT,
                timestamp DATETIME
            )
        """
        )

        # Documents table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT,
                file_type TEXT,
                total_pages INTEGER,
                total_chunks INTEGER,
                upload_date DATETIME,
                metadata TEXT
            )
        """
        )

        # Document chunks table (for tracking individual chunks)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT,
                chunk_index INTEGER,
                content TEXT,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(document_id)
            )
        """
        )

        # Memory stats table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_facts INTEGER,
                total_conversations INTEGER,
                last_updated DATETIME
            )
        """
        )

        conn.commit()
        conn.close()

        logger.info("Initialized database tables")

    def store_fact(
        self,
        category: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        source: str = "user",
        skip_inference: bool = False,
    ) -> str:
        """Store a fact with automatic inference generation"""

        fact_id = hashlib.md5(f"{category}:{key}".encode()).hexdigest()

        # Prepare metadata for ChromaDB (must be flat)
        chromadb_metadata = {
            "category": category,
            "key": key,
            "value": value[:500] if len(value) > 500 else value,
            "confidence": confidence,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }

        # Store in ChromaDB for semantic search
        self.facts_collection.upsert(
            ids=[fact_id],
            documents=[f"{category}: {key} is {value}"],
            metadatas=[chromadb_metadata],
        )

        # Store in SQLite for structured queries
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO facts
            (id, category, key, value, confidence, source, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                fact_id,
                category,
                key,
                value,
                confidence,
                source,
                datetime.now(),
                json.dumps({"original": True}),
            ),
        )

        conn.commit()
        conn.close()

        # Generate inferences (only for user-provided facts, not inferred ones)
        if not skip_inference and source != "inference":
            self._generate_inferences({f"{category}_{key}": value})

        logger.info(
            f"Stored fact: {category}:{key} = {value[:100] if len(value) > 100 else value}"
        )
        return fact_id

    def _generate_inferences(self, new_facts: Dict[str, str]):
        """Generate and store inferences from new facts"""

        all_facts = self.get_all_facts()
        all_facts.update(new_facts)

        inferences = self.inference_engine.infer(all_facts)

        for inference in inferences:
            self.store_fact(
                category="inferred",
                key=inference["type"],
                value=inference["value"],
                confidence=inference["confidence"],
                source="inference",
            )

            logger.info(
                f"Generated inference: {inference['type']} = {inference['value']}"
            )

    def store_conversation(self, query: str, response: str, metadata: Dict = None):
        """Store a conversation exchange with sanitized metadata"""

        conv_id = hashlib.md5(
            f"{query}:{datetime.now().isoformat()}".encode()
        ).hexdigest()

        # Sanitize metadata for ChromaDB
        safe_metadata = {
            "query": query[:500] if len(query) > 500 else query,
            "response": response[:500] if len(response) > 500 else response,
            "timestamp": datetime.now().isoformat(),
        }

        # Add selected fields from metadata if they exist
        if metadata:
            if "selected_model" in metadata:
                safe_metadata["selected_model"] = str(metadata["selected_model"])
            if "total_ensemble_time" in metadata:
                safe_metadata["total_time"] = float(metadata["total_ensemble_time"])
            if "successful_models" in metadata:
                safe_metadata["successful_models"] = int(metadata["successful_models"])
            if "total_models" in metadata:
                safe_metadata["total_models"] = int(metadata["total_models"])
            if "used_web_search" in metadata:
                safe_metadata["used_web_search"] = bool(metadata["used_web_search"])

            # Store complex scores as JSON string
            if "all_scores" in metadata:
                safe_metadata["scores_json"] = json.dumps(metadata["all_scores"])

        # Store in ChromaDB
        self.conversations_collection.add(
            ids=[conv_id],
            documents=[f"User: {query}\nAssistant: {response}"],
            metadatas=[safe_metadata],
        )

        # Store full conversation in SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO conversations (id, query, response, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                conv_id,
                query,
                response,
                json.dumps(metadata) if metadata else "{}",
                datetime.now(),
            ),
        )

        conn.commit()
        conn.close()

        logger.debug(f"Stored conversation: {conv_id}")
        return conv_id

    def search_facts(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search facts using semantic similarity"""

        try:
            results = self.facts_collection.query(
                query_texts=[query], n_results=n_results
            )

            facts = []
            if results["metadatas"] and results["metadatas"][0]:
                for metadata, distance in zip(
                    results["metadatas"][0], results["distances"][0]
                ):
                    facts.append({**metadata, "relevance": 1 - distance})

            return facts
        except Exception as e:
            logger.error(f"Error searching facts: {e}")
            return []

    def search_conversations(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search past conversations"""

        try:
            results = self.conversations_collection.query(
                query_texts=[query], n_results=n_results
            )

            conversations = []
            if results["metadatas"] and results["metadatas"][0]:
                for metadata, document in zip(
                    results["metadatas"][0], results["documents"][0]
                ):
                    # Reconstruct scores if they were stored as JSON
                    if "scores_json" in metadata:
                        try:
                            metadata["scores"] = json.loads(metadata["scores_json"])
                            del metadata["scores_json"]
                        except:
                            pass

                    conversations.append({"content": document, "metadata": metadata})

            return conversations
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []

    def store_document(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        total_pages: int,
        total_chunks: int,
        chunks: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a processed document with all its chunks

        Args:
            document_id: Unique document identifier
            filename: Original filename
            file_type: File extension (.pdf, .docx, etc.)
            total_pages: Number of pages in document
            total_chunks: Number of chunks created
            chunks: List of chunk dictionaries with 'content' and 'metadata'
            metadata: Additional document-level metadata

        Returns:
            Document ID
        """
        # Store document metadata in SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO documents
            (document_id, filename, file_type, total_pages, total_chunks, upload_date, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                document_id,
                filename,
                file_type,
                total_pages,
                total_chunks,
                datetime.now(),
                json.dumps(metadata) if metadata else "{}",
            ),
        )

        # Store each chunk
        chunk_ids = []
        for chunk in chunks:
            chunk_id = f"{document_id}_chunk_{chunk['chunk_index']}"
            chunk_content = chunk["content"]
            chunk_metadata = chunk.get("metadata", {})

            # Sanitize metadata for ChromaDB
            safe_metadata = sanitize_metadata(chunk_metadata)

            # Store in ChromaDB for semantic search
            self.documents_collection.upsert(
                ids=[chunk_id],
                documents=[chunk_content],
                metadatas=[safe_metadata],
            )

            # Store in SQLite for structured queries
            cursor.execute(
                """
                INSERT OR REPLACE INTO document_chunks
                (chunk_id, document_id, chunk_index, content, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    chunk_id,
                    document_id,
                    chunk["chunk_index"],
                    chunk_content,
                    json.dumps(chunk_metadata),
                ),
            )

            chunk_ids.append(chunk_id)

        conn.commit()
        conn.close()

        logger.info(
            f"Stored document {filename}: {total_chunks} chunks, "
            f"{total_pages} pages, ID: {document_id}"
        )

        return document_id

    def search_documents(
        self, query: str, n_results: int = 5, min_relevance: float = 0.3
    ) -> List[Dict]:
        """
        Search document chunks using semantic similarity

        Args:
            query: Search query
            n_results: Maximum number of results
            min_relevance: Minimum relevance score (0-1)

        Returns:
            List of relevant document chunks with metadata
        """
        try:
            results = self.documents_collection.query(
                query_texts=[query], n_results=n_results
            )

            chunks = []
            if results["metadatas"] and results["metadatas"][0]:
                for metadata, document, distance in zip(
                    results["metadatas"][0],
                    results["documents"][0],
                    results["distances"][0],
                ):
                    relevance = 1 - distance

                    # Only return results above relevance threshold
                    if relevance >= min_relevance:
                        chunks.append(
                            {
                                "content": document,
                                "metadata": metadata,
                                "relevance": relevance,
                            }
                        )

            return chunks

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def get_all_documents(self) -> List[Dict]:
        """Get list of all uploaded documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT document_id, filename, file_type, total_pages,
                   total_chunks, upload_date, metadata
            FROM documents
            ORDER BY upload_date DESC
        """
        )

        documents = []
        for row in cursor.fetchall():
            doc_id, filename, file_type, pages, chunks, upload_date, metadata = row
            documents.append(
                {
                    "document_id": doc_id,
                    "filename": filename,
                    "file_type": file_type,
                    "total_pages": pages,
                    "total_chunks": chunks,
                    "upload_date": upload_date,
                    "metadata": json.loads(metadata) if metadata else {},
                }
            )

        conn.close()
        return documents

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get chunk IDs to delete from ChromaDB
            cursor.execute(
                "SELECT chunk_id FROM document_chunks WHERE document_id = ?",
                (document_id,),
            )
            chunk_ids = [row[0] for row in cursor.fetchall()]

            # Delete from ChromaDB
            if chunk_ids:
                self.documents_collection.delete(ids=chunk_ids)

            # Delete from SQLite
            cursor.execute(
                "DELETE FROM document_chunks WHERE document_id = ?", (document_id,)
            )
            cursor.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

            conn.commit()
            conn.close()

            logger.info(f"Deleted document {document_id} and {len(chunk_ids)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    def get_all_facts(self) -> Dict[str, str]:
        """Get all facts as a dictionary"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT category, key, value FROM facts WHERE confidence > 0.5")
        rows = cursor.fetchall()

        facts = {}
        for category, key, value in rows:
            facts[f"{category}_{key}"] = value

        conn.close()
        return facts

    def _count_tokens(self, text: str) -> int:
        """Rough token count estimation (1 token ≈ 4 chars)"""
        return len(text) // 4

    def get_user_context(self, query: str) -> str:
        """Get relevant user context for a query, including documents"""

        if not MEMORY_CONFIG.get("optimize_context", True):
            # Legacy verbose format
            return self._get_user_context_legacy(query)

        # Optimized format
        return self._get_user_context_optimized(query)

    def _get_user_context_optimized(self, query: str) -> str:
        """Optimized memory context with token limits and relevance filtering"""

        context_parts = []
        current_tokens = 0
        max_tokens = MEMORY_CONFIG.get("max_context_tokens", 150)
        max_preview = MEMORY_CONFIG.get("max_content_preview", 200)

        # Search relevant facts (strict relevance filter)
        max_facts = MEMORY_CONFIG.get("max_facts", 3)
        min_fact_relevance = MEMORY_CONFIG.get("min_fact_relevance", 0.6)

        relevant_facts = self.search_facts(query, n_results=max_facts)
        filtered_facts = [
            f for f in relevant_facts
            if f.get("relevance", 0) >= min_fact_relevance
        ]

        if filtered_facts:
            # Compact format: "Facts: key1=val1, key2=val2"
            fact_items = []
            for fact in filtered_facts[:max_facts]:
                key = fact.get('key', 'fact')
                value = fact.get('value', 'unknown')
                # Truncate long values
                if len(value) > 50:
                    value = value[:47] + "..."
                fact_items.append(f"{key}={value}")

            fact_str = "Facts: " + ", ".join(fact_items)
            fact_tokens = self._count_tokens(fact_str)

            if current_tokens + fact_tokens <= max_tokens:
                context_parts.append(fact_str)
                current_tokens += fact_tokens

        # Search relevant document chunks (if not over budget)
        if current_tokens < max_tokens * 0.8:  # Reserve 20% for potential conversations
            max_docs = MEMORY_CONFIG.get("max_documents", 2)
            min_doc_relevance = MEMORY_CONFIG.get("min_document_relevance", 0.4)

            doc_chunks = self.search_documents(query, n_results=max_docs, min_relevance=min_doc_relevance)

            if doc_chunks:
                # Compact format: "Docs: [file1] content... | [file2] content..."
                doc_items = []
                for chunk in doc_chunks[:max_docs]:
                    filename = chunk["metadata"].get("filename", "Doc")
                    # Shorten filename
                    if len(filename) > 20:
                        filename = filename[:17] + "..."

                    content = chunk["content"][:max_preview]
                    if len(chunk["content"]) > max_preview:
                        content += "..."

                    doc_items.append(f"[{filename}] {content}")

                doc_str = "Docs: " + " | ".join(doc_items)
                doc_tokens = self._count_tokens(doc_str)

                # Only add if within budget
                if current_tokens + doc_tokens <= max_tokens:
                    context_parts.append(doc_str)
                    current_tokens += doc_tokens
                elif current_tokens < max_tokens * 0.5:
                    # Truncate to fit budget
                    available = (max_tokens - current_tokens) * 4  # chars
                    doc_str_truncated = doc_str[:available] + "..."
                    context_parts.append(doc_str_truncated)
                    current_tokens = max_tokens

        # Search relevant past conversations (if budget allows)
        if current_tokens < max_tokens * 0.9:  # Only if we have room
            max_convs = MEMORY_CONFIG.get("max_conversations", 1)
            min_conv_relevance = MEMORY_CONFIG.get("min_conversation_relevance", 0.5)
            recent_days = MEMORY_CONFIG.get("recent_conversation_days", 7)

            past_convs = self.search_conversations(query, n_results=max_convs)

            for conv in past_convs:
                timestamp = conv["metadata"].get("timestamp", "")
                if timestamp:
                    try:
                        timestamp_dt = datetime.fromisoformat(timestamp)
                        days_ago = (datetime.now() - timestamp_dt).days

                        if days_ago < recent_days:
                            query_preview = conv["metadata"].get("query", "")[:80]
                            if query_preview:
                                conv_str = f"Prev: {query_preview}..."
                                conv_tokens = self._count_tokens(conv_str)

                                if current_tokens + conv_tokens <= max_tokens:
                                    context_parts.append(conv_str)
                                    current_tokens += conv_tokens
                    except:
                        pass

        result = " | ".join(context_parts) if context_parts else ""

        if result:
            logger.info(f"Memory context: ~{current_tokens} tokens")

        return result

    def _get_user_context_legacy(self, query: str) -> str:
        """Legacy verbose format (for backward compatibility)"""

        context_parts = []

        # Search relevant facts
        relevant_facts = self.search_facts(query, n_results=5)

        if relevant_facts:
            fact_strs = []
            for fact in relevant_facts:
                if fact.get("relevance", 0) > 0.5:
                    fact_strs.append(
                        f"- {fact.get('key', 'fact')}: {fact.get('value', 'unknown')}"
                    )

            if fact_strs:
                context_parts.append("Known facts about user:\n" + "\n".join(fact_strs))

        # Check if query is asking about documents themselves (metadata queries)
        query_lower = query.lower()
        document_keywords = ['document', 'file', 'uploaded', 'pdf', 'docx', 'last uploaded', 'recent', 'latest']
        is_document_metadata_query = any(keyword in query_lower for keyword in document_keywords)

        # Get all documents
        all_docs = self.get_all_documents()

        # If asking about documents themselves, OR if there are only 1-2 documents (proactively include them)
        should_include_doc_list = is_document_metadata_query or (len(all_docs) > 0 and len(all_docs) <= 2)

        if should_include_doc_list and all_docs:
            context_parts.append("\n=== UPLOADED DOCUMENTS ===")
            for doc in all_docs[:5]:  # Show up to 5 most recent
                context_parts.append(
                    f"\n📄 {doc['filename']}"
                    f"\n   - Uploaded: {doc['upload_date']}"
                    f"\n   - Pages: {doc['total_pages']}, Chunks: {doc['total_chunks']}"
                    f"\n   - Type: {doc['file_type']}"
                )

            # For summarization requests, include actual content from the most recent document
            if 'summarize' in query_lower or 'summary' in query_lower or 'about' in query_lower:
                latest_doc = all_docs[0]
                doc_id = latest_doc['document_id']

                # Get first few chunks to provide content for summarization
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT content FROM document_chunks
                        WHERE document_id = ?
                        ORDER BY chunk_index
                        LIMIT 5
                        """,
                        (doc_id,)
                    )
                    chunks_content = cursor.fetchall()
                    conn.close()

                    if chunks_content:
                        context_parts.append(f"\n\n=== CONTENT FROM {latest_doc['filename']} ===")
                        for idx, (content,) in enumerate(chunks_content, 1):
                            preview = content[:500] if len(content) > 500 else content
                            context_parts.append(f"\n[Section {idx}]\n{preview}")
                            if len(content) > 500:
                                context_parts.append("...")
                except Exception as e:
                    logger.error(f"Error fetching document content: {e}")

        # Search relevant document chunks (semantic search)
        doc_chunks = self.search_documents(query, n_results=5, min_relevance=0.3)

        if doc_chunks and not is_document_metadata_query:  # Only show if not already showing full docs
            context_parts.append("\nRelevant information from documents:")
            for chunk in doc_chunks:
                filename = chunk["metadata"].get("filename", "Unknown")
                chunk_idx = chunk["metadata"].get("chunk_index", 0)
                relevance = chunk.get("relevance", 0)

                # Show snippet of content
                content_preview = chunk["content"][:400]
                if len(chunk["content"]) > 400:
                    content_preview += "..."

                context_parts.append(
                    f"\n[From {filename}, section {chunk_idx + 1}, relevance: {relevance:.2f}]\n"
                    f"{content_preview}"
                )

        # Search relevant past conversations
        past_convs = self.search_conversations(query, n_results=2)

        if past_convs:
            context_parts.append("\nRelevant past conversations:")
            for conv in past_convs:
                timestamp = conv["metadata"].get("timestamp", "")
                if timestamp:
                    try:
                        timestamp_dt = datetime.fromisoformat(timestamp)
                        if (datetime.now() - timestamp_dt).days < 7:
                            query_preview = conv["metadata"].get("query", "")[:100]
                            if query_preview:
                                context_parts.append(f"- {query_preview}...")
                    except:
                        pass

        return "\n".join(context_parts) if context_parts else ""

    def forget(self, category: str = None, key: str = None):
        """Forget specific facts"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if category and key:
            # Find the fact ID to remove from ChromaDB
            cursor.execute(
                "SELECT id FROM facts WHERE category = ? AND key = ?", (category, key)
            )
            result = cursor.fetchone()
            if result:
                fact_id = result[0]
                try:
                    self.facts_collection.delete(ids=[fact_id])
                except:
                    pass

            cursor.execute(
                "DELETE FROM facts WHERE category = ? AND key = ?", (category, key)
            )
            logger.info(f"Forgot {category}:{key}")
        elif category:
            # Find all fact IDs in category
            cursor.execute("SELECT id FROM facts WHERE category = ?", (category,))
            fact_ids = [row[0] for row in cursor.fetchall()]
            if fact_ids:
                try:
                    self.facts_collection.delete(ids=fact_ids)
                except:
                    pass

            cursor.execute("DELETE FROM facts WHERE category = ?", (category,))
            logger.info(f"Forgot all facts in category: {category}")
        else:
            # Clear everything
            try:
                # Get all IDs and delete from ChromaDB
                all_facts = self.facts_collection.get()
                if all_facts["ids"]:
                    self.facts_collection.delete(ids=all_facts["ids"])

                all_convs = self.conversations_collection.get()
                if all_convs["ids"]:
                    self.conversations_collection.delete(ids=all_convs["ids"])
            except:
                pass

            cursor.execute("DELETE FROM facts")
            cursor.execute("DELETE FROM conversations")
            logger.info("Forgot all facts and conversations")

        conn.commit()
        conn.close()


class MemoryManager:
    """Main memory management interface"""

    def __init__(self, memory_dir: str = "memory_store"):
        self.semantic_memory = SemanticMemory(memory_dir)
        self.logger = logger

        # Patterns for extracting facts from natural language
        self.fact_patterns = [
            # Identity
            (r"my name is (\w+)", "identity", "name"),
            (r"i am (\w+)", "identity", "name"),
            (r"call me (\w+)", "identity", "name"),
            # Location
            (r"i live in ([\w\s]+)", "location", "city"),
            (r"i'm from ([\w\s]+)", "location", "origin"),
            (r"i work at ([\w\s]+)", "work", "company"),
            # Preferences
            (r"i (like|love|enjoy) ([\w\s]+)", "preference", "likes"),
            (r"i (hate|dislike) ([\w\s]+)", "preference", "dislikes"),
        ]

    def process_input(self, text: str) -> List[MemoryEntry]:
        """Extract and store facts from user input"""

        import re

        extracted_memories = []

        text_lower = text.lower()

        # Check for explicit memory commands
        if "remember that" in text_lower or "remember:" in text_lower:
            fact_text = text.split("remember that")[-1].split("remember:")[-1].strip()

            fact_id = self.semantic_memory.store_fact(
                category="explicit",
                key="statement",
                value=fact_text,
                source="user_explicit",
            )

            extracted_memories.append(
                MemoryEntry(
                    id=fact_id,
                    type=MemoryType.FACT,
                    content=fact_text,
                    metadata={"explicit": True},
                    timestamp=datetime.now(),
                )
            )

        # Extract facts using patterns
        for pattern, category, key in self.fact_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                value = match[1] if isinstance(match, tuple) else match

                fact_id = self.semantic_memory.store_fact(
                    category=category, key=key, value=value
                )

                extracted_memories.append(
                    MemoryEntry(
                        id=fact_id,
                        type=MemoryType.FACT,
                        content=f"{key}: {value}",
                        metadata={"category": category},
                        timestamp=datetime.now(),
                    )
                )

        # Check for forget commands
        if "forget" in text_lower:
            if "forget everything" in text_lower:
                self.semantic_memory.forget()
                self.logger.info("Forgot all memories")
            elif "forget my name" in text_lower:
                self.semantic_memory.forget("identity", "name")

        return extracted_memories

    def enhance_prompt(self, prompt: str) -> str:
        """Enhance a prompt with relevant memory context"""

        context = self.semantic_memory.get_user_context(prompt)

        if context:
            if MEMORY_CONFIG.get("compact_formatting", True):
                # Optimized compact format
                enhanced_prompt = f"[Memory] {context}\n\nQ: {prompt}"
            else:
                # Legacy verbose format
                enhanced_prompt = f"""User Context:
{context}

Current Query: {prompt}

Please consider the user context when responding. Use any relevant information to personalize your response."""
            return enhanced_prompt

        return prompt

    def save_conversation(self, query: str, response: str, metadata: Dict = None):
        """Save a conversation to memory"""

        # Extract any facts from the query
        self.process_input(query)

        # Store the conversation with sanitized metadata
        self.semantic_memory.store_conversation(query, response, metadata)

    def get_memory_stats(self) -> Dict:
        """Get statistics about stored memories"""

        try:
            conn = sqlite3.connect(self.semantic_memory.db_path)
            cursor = conn.cursor()

            # Count facts
            cursor.execute("SELECT COUNT(*) FROM facts")
            fact_count = cursor.fetchone()[0]

            # Count inferred facts
            cursor.execute('SELECT COUNT(*) FROM facts WHERE source = "inference"')
            inference_count = cursor.fetchone()[0]

            # Count conversations
            cursor.execute("SELECT COUNT(*) FROM conversations")
            conv_count = cursor.fetchone()[0]

            conn.close()

            return {
                "total_facts": fact_count,
                "inferred_facts": inference_count,
                "total_conversations": conv_count,
                "memory_location": str(self.semantic_memory.memory_dir),
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {
                "total_facts": 0,
                "inferred_facts": 0,
                "total_conversations": 0,
                "memory_location": str(self.semantic_memory.memory_dir),
                "error": str(e),
            }

    def export_memories(self) -> Dict:
        """Export all memories as JSON (for backup)"""

        facts = self.semantic_memory.get_all_facts()

        return {
            "facts": facts,
            "exported_at": datetime.now().isoformat(),
            "stats": self.get_memory_stats(),
        }
