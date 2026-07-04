"""
Data models for code chunks.

Chunks are semantic units (functions, classes, modules) not arbitrary text splits.
This is the key differentiator from traditional RAG.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from app.ingestion import Language


@dataclass
class CodeChunk:
    """
    Represents a semantic unit of code (function, class, module).
    
    Key differentiator: Chunks correspond to code structure, not arbitrary text.
    
    Traditional RAG:
        Chunk 1: characters 0-512 (arbitrary, might split mid-function)
        Chunk 2: characters 513-1024 (arbitrary)
    
    Our approach:
        Chunk 1: def create_access_token() - complete function
        Chunk 2: class UserService - complete class with methods
        Chunk 3: import statements block
    
    Attributes:
        text: The actual code content (complete function/class)
        chunk_type: Type of chunk ("function", "class", "import", "module")
        name: Name of the function/class/module
        file_path: Relative path to source file
        repository_name: Repository this chunk belongs to
        language: Programming language
        line_start: Starting line number (for jump-to-definition)
        line_end: Ending line number
        metadata: Additional metadata (params, decorators, etc.)
    """
    
    text: str
    chunk_type: str  # "function", "class", "import", "module", "method"
    name: str
    file_path: str
    repository_name: str
    language: Language
    line_start: int
    line_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Embeddings (added by Embedder stage)
    embedding: Optional[List[float]] = None
    
    # Retrieval metadata (added by Retriever stage)
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    combined_score: Optional[float] = None
    
    def __post_init__(self):
        """Ensure language is Language enum."""
        if isinstance(self.language, str):
            from app.ingestion import Language
            self.language = Language.from_string(self.language)
    
    @property
    def line_count(self) -> int:
        """Get number of lines in this chunk."""
        return self.line_end - self.line_start + 1
    
    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.text)
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for storage/serialization.
        
        Returns:
            dict: Chunk data suitable for ChromaDB or JSON
        """
        return {
            "text": self.text,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "file_path": self.file_path,
            "repository_name": self.repository_name,
            "language": self.language.value,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "line_count": self.line_count,
            "char_count": self.char_count,
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"CodeChunk({self.chunk_type}:{self.name}, {self.file_path}:{self.line_start}-{self.line_end})"
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"CodeChunk(type={self.chunk_type!r}, "
            f"name={self.name!r}, "
            f"file={self.file_path!r}, "
            f"lines={self.line_start}-{self.line_end})"
        )


@dataclass
class ChunkingResult:
    """
    Result of chunking a CodeFile.
    
    Contains statistics and the generated chunks.
    """
    
    file_path: str
    repository_name: str
    language: Language
    chunks: List[CodeChunk]
    
    # Statistics
    total_chunks: int = 0
    function_chunks: int = 0
    class_chunks: int = 0
    import_chunks: int = 0
    other_chunks: int = 0
    
    def __post_init__(self):
        """Calculate statistics."""
        self.total_chunks = len(self.chunks)
        self.function_chunks = sum(1 for c in self.chunks if c.chunk_type == "function")
        self.class_chunks = sum(1 for c in self.chunks if c.chunk_type == "class")
        self.import_chunks = sum(1 for c in self.chunks if c.chunk_type == "import")
        self.other_chunks = self.total_chunks - (
            self.function_chunks + self.class_chunks + self.import_chunks
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "repository_name": self.repository_name,
            "language": self.language.value,
            "total_chunks": self.total_chunks,
            "function_chunks": self.function_chunks,
            "class_chunks": self.class_chunks,
            "import_chunks": self.import_chunks,
            "other_chunks": self.other_chunks,
            "chunks": [c.to_dict() for c in self.chunks],
        }
