"""
Code chunk indexer - persistence layer for the RAG pipeline.

SINGLE RESPONSIBILITY: Persist embedded CodeChunk objects into vector store.

NOT responsible for:
- Embedding (that's CodeEmbedder)
- Retrieval (that's Retriever)
- Answer generation (that's LLM Chain)

Only responsible for:
- Validating chunks have embeddings
- Preparing consistent metadata
- Persisting to vector store
- Incremental updates
- Statistics and logging
"""
from typing import List, Dict, Any, Optional
import hashlib
import time
from dataclasses import dataclass

from app.core import logger, settings
from app.ingestion import CodeFile
from app.rag.chunk_models import CodeChunk
from app.rag.vector_store import VectorStore, ChromaVectorStore


@dataclass
class IndexingStats:
    """Statistics from an indexing operation."""
    
    total_chunks: int = 0
    indexed_chunks: int = 0
    skipped_chunks: int = 0  # Already exist
    failed_chunks: int = 0
    indexing_time_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_chunks": self.total_chunks,
            "indexed_chunks": self.indexed_chunks,
            "skipped_chunks": self.skipped_chunks,
            "failed_chunks": self.failed_chunks,
            "indexing_time_seconds": self.indexing_time_seconds,
        }


class CodeIndexer:
    """
    Persistence layer for embedded code chunks.
    
    SINGLE RESPONSIBILITY: Store embedded chunks in vector database.
    
    Features:
    - Validates chunks have embeddings
    - Prepares consistent metadata
    - Batch inserts for efficiency
    - Incremental indexing (only index changed chunks)
    - Supports multiple vector stores (ChromaDB, Milvus, etc.)
    - Comprehensive statistics and logging
    
    NOT responsible for embedding or retrieval.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        batch_size: int = 100,
        skip_existing: bool = True
    ):
        """
        Initialize code indexer.
        
        Args:
            vector_store: Vector store implementation (defaults to ChromaDB)
            batch_size: Batch size for inserts
            skip_existing: Whether to skip chunks that already exist
        """
        self.vector_store = vector_store or self._create_default_vector_store()
        self.batch_size = batch_size
        self.skip_existing = skip_existing
        
        logger.info(
            f"CodeIndexer initialized: "
            f"store={self.vector_store.__class__.__name__}, "
            f"batch_size={batch_size}, "
            f"skip_existing={skip_existing}"
        )
    
    def index_file(self, code_file: CodeFile) -> IndexingStats:
        """
        Index all chunks from a CodeFile.
        
        Args:
            code_file: CodeFile with embedded chunks
            
        Returns:
            IndexingStats: Statistics about the indexing operation
        """
        stats = IndexingStats()
        start_time = time.time()
        
        # Validate file has chunks
        if not code_file.is_chunked():
            logger.warning(f"File not chunked: {code_file.relative_path_str}")
            stats.indexing_time_seconds = time.time() - start_time
            return stats
        
        # Validate file has embeddings
        if not code_file.is_embedded():
            logger.warning(f"File not embedded: {code_file.relative_path_str}")
            stats.indexing_time_seconds = time.time() - start_time
            return stats
        
        logger.info(f"Indexing {code_file.relative_path_str}")
        
        try:
            # Convert chunk dicts to CodeChunk objects
            chunks = code_file.chunks
            
            # Index chunks
            chunk_stats = self.index_chunks(chunks)
            
            # Update stats
            stats.total_chunks = chunk_stats.total_chunks
            stats.indexed_chunks = chunk_stats.indexed_chunks
            stats.skipped_chunks = chunk_stats.skipped_chunks
            stats.failed_chunks = chunk_stats.failed_chunks
            stats.indexing_time_seconds = time.time() - start_time
            
            logger.info(
                f"Indexed {code_file.relative_path_str}: "
                f"{stats.indexed_chunks}/{stats.total_chunks} chunks "
                f"({stats.skipped_chunks} skipped, {stats.failed_chunks} failed) "
                f"in {stats.indexing_time_seconds:.2f}s"
            )
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to index {code_file.relative_path_str}: {e}")
            stats.indexing_time_seconds = time.time() - start_time
            return stats
    
    def index_files(self, code_files: List[CodeFile]) -> IndexingStats:
        """
        Index all chunks from multiple CodeFiles.
        
        Args:
            code_files: List of CodeFiles with embedded chunks
            
        Returns:
            IndexingStats: Aggregated statistics
        """
        total_stats = IndexingStats()
        start_time = time.time()
        
        for code_file in code_files:
            file_stats = self.index_file(code_file)
            total_stats.total_chunks += file_stats.total_chunks
            total_stats.indexed_chunks += file_stats.indexed_chunks
            total_stats.skipped_chunks += file_stats.skipped_chunks
            total_stats.failed_chunks += file_stats.failed_chunks
        
        total_stats.indexing_time_seconds = time.time() - start_time
        
        logger.info(
            f"Batch indexing complete: "
            f"{total_stats.indexed_chunks}/{total_stats.total_chunks} chunks indexed "
            f"({total_stats.skipped_chunks} skipped, {total_stats.failed_chunks} failed) "
            f"in {total_stats.indexing_time_seconds:.2f}s"
        )
        
        return total_stats
    
    def index_chunks(self, chunks: List[CodeChunk]) -> IndexingStats:
        """
        Index a list of CodeChunk objects.
        
        This is the core indexing method. It:
        1. Validates chunks have embeddings
        2. Generates stable chunk IDs
        3. Prepares metadata dictionaries
        4. Checks for existing chunks (if skip_existing=True)
        5. Batch inserts into vector store
        
        Args:
            chunks: List of CodeChunk objects with embeddings
            
        Returns:
            IndexingStats: Statistics about the operation
        """
        stats = IndexingStats()
        stats.total_chunks = len(chunks)
        start_time = time.time()
        
        if not chunks:
            logger.warning("No chunks to index")
            return stats
        
        logger.info(f"Indexing {len(chunks)} chunks")
        
        # Prepare chunks for indexing
        ids_batch = []
        embeddings_batch = []
        documents_batch = []
        metadatas_batch = []
        
        for chunk in chunks:
            # Validate chunk has embedding
            if not self._validate_chunk(chunk):
                stats.failed_chunks += 1
                continue
            
            # Generate stable chunk ID
            chunk_id = self._generate_chunk_id(chunk)
            
            # Check if chunk exists (for incremental indexing)
            if self.skip_existing and self.vector_store.document_exists(chunk_id):
                logger.debug(f"Skipping existing chunk: {chunk_id}")
                stats.skipped_chunks += 1
                continue
            
            # Prepare for indexing
            ids_batch.append(chunk_id)
            embeddings_batch.append(chunk.embedding)
            documents_batch.append(chunk.text)
            metadatas_batch.append(self._prepare_metadata(chunk))
            
            # Batch insert when batch is full
            if len(ids_batch) >= self.batch_size:
                success = self._insert_batch(
                    ids_batch,
                    embeddings_batch,
                    documents_batch,
                    metadatas_batch
                )
                if success:
                    stats.indexed_chunks += len(ids_batch)
                else:
                    stats.failed_chunks += len(ids_batch)
                
                # Clear batch
                ids_batch = []
                embeddings_batch = []
                documents_batch = []
                metadatas_batch = []
        
        # Insert remaining chunks
        if ids_batch:
            success = self._insert_batch(
                ids_batch,
                embeddings_batch,
                documents_batch,
                metadatas_batch
            )
            if success:
                stats.indexed_chunks += len(ids_batch)
            else:
                stats.failed_chunks += len(ids_batch)
        
        stats.indexing_time_seconds = time.time() - start_time
        
        logger.info(
            f"Indexed {stats.indexed_chunks}/{stats.total_chunks} chunks "
            f"({stats.skipped_chunks} skipped, {stats.failed_chunks} failed) "
            f"in {stats.indexing_time_seconds:.2f}s"
        )
        
        return stats
    
    def _validate_chunk(self, chunk: CodeChunk) -> bool:
        """
        Validate that a chunk is ready for indexing.
        
        Args:
            chunk: CodeChunk to validate
            
        Returns:
            bool: True if valid
        """
        # Must have embedding
        if not chunk.embedding or len(chunk.embedding) == 0:
            logger.warning(f"Chunk missing embedding: {chunk.name}")
            return False
        
        # Must have text
        if not chunk.text or len(chunk.text.strip()) == 0:
            logger.warning(f"Chunk has empty text: {chunk.name}")
            return False
        
        return True
    
    def _generate_chunk_id(self, chunk: CodeChunk) -> str:
        """
        Generate a stable, unique ID for a chunk.
        
        Uses: repository + file_path + chunk_type + name + line_start
        This ensures the same chunk always gets the same ID.
        
        Args:
            chunk: CodeChunk
            
        Returns:
            str: Stable chunk ID
        """
        id_string = (
            f"{chunk.repository_name}/"
            f"{chunk.file_path}/"
            f"{chunk.chunk_type}/"
            f"{chunk.name}/"
            f"{chunk.line_start}"
        )
        
        # Hash for shorter, consistent IDs
        id_hash = hashlib.sha256(id_string.encode()).hexdigest()[:16]
        return f"chunk_{id_hash}"
    
    def _prepare_metadata(self, chunk: CodeChunk) -> Dict[str, Any]:
        """
        Prepare consistent metadata dictionary for storage.
        
        This is the metadata that will be stored with the chunk and
        used for filtering during retrieval.
        
        Args:
            chunk: CodeChunk
            
        Returns:
            dict: Metadata dictionary
        """
        metadata = {
            # Core identifiers
            "repository_id": self._hash_repository(chunk.repository_name), 
            "repository_name": chunk.repository_name,
            "file_path": chunk.file_path,
            "language": chunk.language.value,
            
            # Chunk information
            "chunk_type": chunk.chunk_type,
            "chunk_name": chunk.name,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "line_count": chunk.line_count,
            "char_count": chunk.char_count,
            
            # Embedding information
            "embedding_model": settings.embedding_model,
            "embedding_dim": len(chunk.embedding),
        }
        
        # Add chunk-specific metadata
        if chunk.metadata:
            # Add function metadata
            if chunk.chunk_type == "function":
                if "params" in chunk.metadata:
                    metadata["params"] = str(chunk.metadata["params"])
                if "return_type" in chunk.metadata:
                    metadata["return_type"] = str(chunk.metadata["return_type"])
                if "decorators" in chunk.metadata:
                    metadata["decorators"] = ",".join(chunk.metadata["decorators"])
            
            # Add class metadata
            elif chunk.chunk_type == "class":
                if "bases" in chunk.metadata:
                    metadata["bases"] = ",".join(chunk.metadata["bases"])
                if "method_names" in chunk.metadata:
                    metadata["method_names"] = ",".join(chunk.metadata["method_names"])
            
            # Add docstring (truncated)
            if "docstring" in chunk.metadata:
                docstring = chunk.metadata["docstring"]
                metadata["docstring"] = docstring[:200]  # Truncate for storage
        # print(metadata["repository_name"])
        # print(self.vector_store.collection.peek())
        print("Repository name :", chunk.repository_name)
        print("Repository id   :", self._hash_repository(chunk.repository_name))
        return metadata
    
    def _insert_batch(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert a batch of chunks into vector store.
        
        Args:
            ids: Chunk IDs
            embeddings: Embedding vectors
            documents: Document texts
            metadatas: Metadata dictionaries
            
        Returns:
            bool: True if successful
        """
        try:
            return self.vector_store.add_documents(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"Failed to insert batch of {len(ids)} chunks: {e}")
            return False
    
    def _create_default_vector_store(self) -> VectorStore:
        """
        Create default vector store (ChromaDB).
        
        Returns:
            VectorStore: ChromaDB vector store
        """
        return ChromaVectorStore(
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
            distance_metric=settings.chroma_distance_metric
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get indexer statistics.
        
        Returns:
            dict: Indexer statistics
        """
        return {
            "vector_store": self.vector_store.__class__.__name__,
            "total_indexed": self.vector_store.count(),
            "batch_size": self.batch_size,
            "skip_existing": self.skip_existing,
        }
    
    def clear_index(self) -> bool:
        """
        Clear all indexed chunks.
        
        WARNING: This deletes all data!
        
        Returns:
            bool: True if successful
        """
        logger.warning("Clearing entire index!")
        return self.vector_store.clear()
