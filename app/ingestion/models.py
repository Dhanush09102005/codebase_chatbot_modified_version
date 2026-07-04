"""
Data models for the ingestion pipeline.

These models are enriched as they flow through the pipeline:
FileScanner → CodeFile → Parser → Chunker → Embedder → Retriever
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, List, Dict
from app.ingestion.language import Language


@dataclass
class CodeFile:
    """
    Represents a code file with metadata that gets enriched through the pipeline.
    
    This is the fundamental data object that flows through the entire RAG pipeline.
    Each stage adds more information without changing the object structure.
    
    Core attributes (from FileScanner):
        absolute_path: Full path to the file on disk
        relative_path: Relative path from repository root (for display/retrieval)
        repository_root: Path to the repository this file belongs to
        repository_name: Name of the repository (e.g., "fastapi", "projecthub")
        language: Programming language (Language enum for type safety)
        extension: File extension (e.g., '.py', '.js')
        size: File size in bytes
        content_hash: SHA-256 hash of file content (for incremental indexing)
        
    Parser enrichment fields (added by Parser stage):
        imports: List of imported modules/packages
        functions: List of function definitions with metadata
        classes: List of class definitions with metadata
        docstrings: Extracted documentation strings
        ast_tree: Abstract syntax tree representation
        
    Chunker enrichment fields (added by Chunker stage):
        chunks: List of code chunks with metadata (line numbers, types, etc.)
        
    Embedder enrichment fields (added by Embedder stage):
        embeddings: Vector embeddings for each chunk
        
    Retriever enrichment fields (used by Retriever):
        metadata: Additional metadata (scores, context, etc.)
    """
    
    # Core attributes (FileScanner stage)
    absolute_path: Path
    relative_path: Path
    repository_root: Path
    repository_name: str
    language: Language
    extension: str
    size: int
    content_hash: str  # SHA-256 for incremental indexing
    
    # Parser enrichment (Parser stage)
    imports: Optional[List[str]] = None
    functions: Optional[List[Dict[str, Any]]] = None
    classes: Optional[List[Dict[str, Any]]] = None
    docstrings: Optional[List[str]] = None
    ast_tree: Optional[Any] = None
    
    # Chunker enrichment (Chunker stage)
    chunks: Optional[List[Dict[str, Any]]] = None
    
    # Embedder enrichment (Embedder stage)
    embeddings: Optional[List[Any]] = None
    
    # Retriever enrichment (Retriever stage)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize paths."""
        self.absolute_path = Path(self.absolute_path)
        self.relative_path = Path(self.relative_path)
        self.repository_root = Path(self.repository_root)
        
        # Ensure language is Language enum
        if isinstance(self.language, str):
            self.language = Language.from_string(self.language)
    
    @property
    def full_path_str(self) -> str:
        """Get full path as string."""
        return str(self.absolute_path)
    
    @property
    def relative_path_str(self) -> str:
        """Get relative path as string (useful for display)."""
        return str(self.relative_path)
    
    @property
    def file_name(self) -> str:
        """Get just the file name."""
        return self.absolute_path.name
    
    @property
    def repo_name(self) -> str:
        """Get repository name."""
        return self.repository_name
    
    @property
    def language_str(self) -> str:
        """Get language as string."""
        return str(self.language.value)
    
    def has_changed(self, previous_hash: str) -> bool:
        """
        Check if file content has changed since last indexing.
        
        Args:
            previous_hash: SHA-256 hash from previous indexing
            
        Returns:
            bool: True if content has changed
        """
        return self.content_hash != previous_hash
    
    def is_parsed(self) -> bool:
        """Check if Parser stage has been applied."""
        return self.ast_tree is not None
    
    def is_chunked(self) -> bool:
        """Check if Chunker stage has been applied."""
        return self.chunks is not None and len(self.chunks) > 0
    
    def is_embedded(self) -> bool:
        """Check if Embedder stage has been applied."""
        return self.embeddings is not None and len(self.embeddings) > 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary (useful for serialization)."""
        return {
            "absolute_path": str(self.absolute_path),
            "relative_path": str(self.relative_path),
            "repository_root": str(self.repository_root),
            "repository_name": self.repository_name,
            "language": self.language.value,
            "extension": self.extension,
            "size": self.size,
            "content_hash": self.content_hash,
            "file_name": self.file_name,
            "has_imports": self.imports is not None,
            "has_functions": self.functions is not None,
            "has_classes": self.classes is not None,
            "has_chunks": self.is_chunked(),
            "has_embeddings": self.is_embedded(),
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"CodeFile({self.relative_path_str}, {self.language.value}, {self.repository_name})"
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"CodeFile(path={self.relative_path_str!r}, "
            f"language={self.language.value!r}, "
            f"repo={self.repository_name!r}, "
            f"size={self.size}, "
            f"hash={self.content_hash[:8]}...)"
        )
