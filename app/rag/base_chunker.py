"""
Base chunker interface and fallback text-based chunker.

The AST chunker is preferred, but this provides a fallback for unsupported languages.
"""
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

from app.core import logger
from app.ingestion import CodeFile
from app.rag.chunk_models import CodeChunk, ChunkingResult


class BaseChunker(ABC):
    """
    Abstract base class for chunkers.
    
    All chunkers follow the progressive enrichment pattern:
    they enrich the CodeFile object rather than creating new objects.
    """
    
    @abstractmethod
    def chunk(self, code_file: CodeFile) -> bool:
        """
        Chunk a code file and enrich it with chunks.
        
        Args:
            code_file: CodeFile to chunk (will be enriched in-place)
            
        Returns:
            bool: True if chunking succeeded
        """
        pass
    
    def chunk_batch(self, code_files: List[CodeFile]) -> int:
        """
        Chunk multiple files in batch.
        
        Args:
            code_files: List of CodeFile objects
            
        Returns:
            int: Number of successfully chunked files
        """
        success_count = 0
        for code_file in code_files:
            if self.chunk(code_file):
                success_count += 1
        
        logger.info(f"Batch chunking complete: {success_count}/{len(code_files)} files")
        return success_count


class TextChunker(BaseChunker):
    """
    Fallback text-based chunker for unsupported languages.
    
    Splits file into fixed-size chunks when AST parsing is unavailable.
    This is a fallback - AST chunking is always preferred.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(f"TextChunker initialized: size={chunk_size}, overlap={overlap}")
    
    def chunk(self, code_file: CodeFile) -> bool:
        """
        Chunk file using simple text splitting.
        
        This is a fallback when AST parsing is not available.
        
        Args:
            code_file: CodeFile to chunk
            
        Returns:
            bool: True if chunking succeeded
        """
        if code_file.is_chunked():
            logger.debug(f"File already chunked: {code_file.relative_path_str}")
            return True
        
        logger.info(f"Text-chunking {code_file.relative_path_str} (fallback)")
        
        try:
            # Read file content
            content = self._read_file(code_file.absolute_path)
            if not content:
                logger.warning(f"Empty file: {code_file.relative_path_str}")
                return False
            
            # Split into chunks
            chunks = []
            start = 0
            chunk_num = 0
            
            while start < len(content):
                end = min(start + self.chunk_size, len(content))
                chunk_text = content[start:end]
                
                # Estimate line numbers (approximate)
                lines_before = content[:start].count('\n')
                lines_in_chunk = chunk_text.count('\n')
                
                chunk = CodeChunk(
                    text=chunk_text,
                    chunk_type="text_block",
                    name=f"chunk_{chunk_num}",
                    file_path=code_file.relative_path_str,
                    repository_name=code_file.repository_name,
                    language=code_file.language,
                    line_start=lines_before + 1,
                    line_end=lines_before + lines_in_chunk + 1,
                    metadata={
                        "chunking_method": "text",
                        "chunk_index": chunk_num,
                    }
                )
                chunks.append(chunk)
                
                # Move to next chunk with overlap
                start += self.chunk_size - self.overlap
                chunk_num += 1
            
            # Enrich CodeFile (progressive enrichment pattern)
            # Enrich CodeFile (progressive enrichment pattern)
            code_file.chunks = chunks
            
            logger.info(
                f"Text-chunked {code_file.relative_path_str}: {len(chunks)} chunks"
            )
            return True
        
        except Exception as e:
            logger.error(f"Failed to text-chunk {code_file.relative_path_str}: {e}")
            return False
    
    def _read_file(self, file_path: Path) -> str:
        """
        Read file content with encoding fallback.
        
        Args:
            file_path: Path to file
            
        Returns:
            str: File content or empty string if read fails
        """
        encodings = ["utf-8", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, Exception):
                continue
        
        logger.error(f"Failed to read file: {file_path}")
        return ""
