"""
Embedding pipeline for code chunks.

This is NOT just a wrapper around model.encode() - it's a pipeline that:
- Accepts CodeChunk objects (not raw strings)
- Prepares text with metadata for better embeddings
- Batches requests efficiently
- Enriches chunks with embeddings (progressive enrichment)
- Handles errors gracefully
"""
from typing import List, Optional
import numpy as np

from app.core import logger, settings
from app.ingestion import CodeFile
from app.rag.chunk_models import CodeChunk
from app.rag.embedding_models import EmbeddingModel, create_embedding_model


class CodeEmbedder:
    """
    Embedding pipeline for code chunks.
    
    Key features:
    - Works with CodeChunk objects (not raw strings)
    - Enriches chunks with embeddings (progressive enrichment)
    - Batch processing for efficiency
    - Prepares text with metadata for better embeddings
    - Multiple embedding model support
    
    This is a pipeline, not just a model wrapper.
    """
    
    def __init__(
        self,
        model: Optional[EmbeddingModel] = None,
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
        include_metadata_in_text: bool = True
    ):
        """
        Initialize code embedder.
        
        Args:
            model: Embedding model (creates default if None)
            batch_size: Batch size for embedding (defaults to settings)
            normalize: Whether to normalize embeddings (defaults to settings)
            include_metadata_in_text: Whether to prepend metadata to chunk text
        """
        self.model = model or create_embedding_model()
        self.batch_size = batch_size or settings.embedding_batch_size
        self.normalize = normalize if normalize is not None else settings.embedding_normalize
        self.include_metadata_in_text = include_metadata_in_text
        
        logger.info(
            f"CodeEmbedder initialized: "
            f"model={self.model.model_name}, "
            f"batch_size={self.batch_size}, "
            f"dim={self.model.embedding_dim}"
        )
    
    def embed_file(self, code_file: CodeFile) -> bool:
        """
        Embed all chunks in a CodeFile.
        
        Enriches the CodeFile object with embeddings (progressive enrichment).
        
        Args:
            code_file: CodeFile with chunks to embed
            
        Returns:
            bool: True if embedding succeeded
        """
        if code_file.is_embedded():
            logger.debug(f"File already embedded: {code_file.relative_path_str}")
            return True
        
        if not code_file.is_chunked():
            logger.warning(f"File not chunked: {code_file.relative_path_str}")
            return False
        
        logger.info(f"Embedding {code_file.relative_path_str}")
        
        try:
            # Convert chunk dicts to CodeChunk objects
            print(code_file.chunks[0])
            chunks = code_file.chunks
            
            # Embed chunks
            self.embed_chunks(chunks)
            
            # Update CodeFile with embeddings
            code_file.embeddings = [chunk.embedding for chunk in chunks]
            
           # Also update CodeChunk objects with embeddings
            for i, chunk in enumerate(chunks):
                code_file.chunks[i].embedding = chunk.embedding
            
            logger.info(
                f"Embedded {code_file.relative_path_str}: "
                f"{len(code_file.embeddings)} embeddings"
            )
            return True
        
        except Exception as e:
            logger.error(f"Failed to embed {code_file.relative_path_str}: {e}")
            return False
    
    def embed_files(self, code_files: List[CodeFile]) -> int:
        """
        Embed all chunks in multiple CodeFiles.
        
        Args:
            code_files: List of CodeFiles to embed
            
        Returns:
            int: Number of successfully embedded files
        """
        success_count = 0
        for code_file in code_files:
            if self.embed_file(code_file):
                success_count += 1
        
        logger.info(f"Batch embedding complete: {success_count}/{len(code_files)} files")
        return success_count
    
    def embed_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """
        Embed a list of CodeChunk objects.
        
        Enriches each chunk with its embedding vector (progressive enrichment).
        
        Args:
            chunks: List of CodeChunk objects to embed
            
        Returns:
            List[CodeChunk]: Same chunks, enriched with embeddings
        """
        if not chunks:
            logger.warning("No chunks to embed")
            return chunks
        
        logger.info(f"Embedding {len(chunks)} chunks")
        
        try:
            # Prepare texts for embedding
            texts = [self._prepare_text(chunk) for chunk in chunks]
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize=self.normalize
            )
            
            # Enrich chunks with embeddings (progressive enrichment)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()  # Convert to list for serialization
            
            logger.info(
                f"Embedded {len(chunks)} chunks: "
                f"dim={self.model.embedding_dim}"
            )
            return chunks
        
        except Exception as e:
            logger.error(f"Failed to embed {len(chunks)} chunks: {e}")
            raise
    
    def _prepare_text(self, chunk: CodeChunk) -> str:
        """
        Prepare chunk text for embedding.
        
        Optionally prepends metadata to improve embedding quality:
        - Chunk type (function, class, method)
        - Chunk name
        - File path
        - Docstring (if available)
        
        This helps the embedding model understand context.
        
        Args:
            chunk: CodeChunk to prepare
            
        Returns:
            str: Prepared text for embedding
        """
        if not self.include_metadata_in_text:
            return chunk.text
        
        # Build metadata prefix
        metadata_parts = []
        
        # Type and name
        metadata_parts.append(f"{chunk.chunk_type}: {chunk.name}")
        
        # File path
        metadata_parts.append(f"File: {chunk.file_path}")
        
        # Language
        metadata_parts.append(f"Language: {chunk.language.value}")
        
        # Docstring (if available and not too long)
        docstring = chunk.metadata.get("docstring")
        if docstring and len(docstring) < 200:
            metadata_parts.append(f"Description: {docstring}")
        
        # Parameters (for functions)
        if chunk.chunk_type == "function" and chunk.metadata.get("params"):
            metadata_parts.append(f"Parameters: {chunk.metadata['params']}")
        
        # Combine metadata with actual code
        metadata_header = "\n".join(metadata_parts)
        prepared_text = f"{metadata_header}\n\n{chunk.text}"
        
        return prepared_text
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.model.embedding_dim
    
    def get_stats(self) -> dict:
        """
        Get embedder statistics.
        
        Returns:
            dict: Statistics about the embedder
        """
        return {
            "model_name": self.model.model_name,
            "embedding_dim": self.model.embedding_dim,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "include_metadata": self.include_metadata_in_text,
        }


class EmbeddingCache:
    """
    Cache for embeddings to avoid recomputing unchanged chunks.
    
    Uses chunk content hash as key.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.cache = {}
        self.max_size = max_size
        logger.info(f"EmbeddingCache initialized: max_size={max_size}")
    
    def get(self, content_hash: str) -> Optional[List[float]]:
        """
        Get cached embedding.
        
        Args:
            content_hash: Hash of chunk content
            
        Returns:
            List[float]: Cached embedding or None
        """
        return self.cache.get(content_hash)
    
    def put(self, content_hash: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.
        
        Args:
            content_hash: Hash of chunk content
            embedding: Embedding vector
        """
        # Simple LRU: remove oldest if at max size
        if len(self.cache) >= self.max_size:
            # Remove first item (oldest)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[content_hash] = embedding
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        logger.info("EmbeddingCache cleared")
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)
