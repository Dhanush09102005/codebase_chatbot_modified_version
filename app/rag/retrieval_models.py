"""
Data models for retrieval results.

These models carry rich metadata through the retrieval pipeline.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from app.ingestion import Language


@dataclass
class RetrievalResult:
    """
    A single retrieval result with rich metadata.
    
    This is NOT just text + score - it carries all the structural
    information we've built (file, function name, line numbers, etc.)
    """
    
    # Document content
    chunk_id: str
    text: str
    
    # Scores (different retrieval methods)
    vector_score: float  # Cosine similarity from vector search
    bm25_score: Optional[float] = None  # BM25 keyword score
    combined_score: Optional[float] = None  # Hybrid score
    rerank_score: Optional[float] = None  # After reranking
    
    # Rich metadata (from indexing)
    repository_name: Optional[str] = None
    file_path: Optional[str] = None
    language: Optional[str] = None
    chunk_type: Optional[str] = None  # function, class, method, import
    chunk_name: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate combined score if not provided."""
        if self.combined_score is None and self.bm25_score is not None:
            # Default hybrid combination
            alpha = 0.5
            self.combined_score = (
                alpha * self.vector_score + 
                (1 - alpha) * self.bm25_score
            )
    
    @property
    def location(self) -> str:
        """Get human-readable location string."""
        if self.file_path and self.line_start:
            return f"{self.file_path}:{self.line_start}"
        elif self.file_path:
            return self.file_path
        return "unknown"
    
    @property
    def context_summary(self) -> str:
        """Get context summary for display."""
        parts = []
        
        if self.chunk_type:
            parts.append(self.chunk_type)
        if self.chunk_name:
            parts.append(self.chunk_name)
        if self.language:
            parts.append(f"({self.language})")
        
        return " ".join(parts) if parts else "code chunk"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "combined_score": self.combined_score,
            "rerank_score": self.rerank_score,
            "repository_name": self.repository_name,
            "file_path": self.file_path,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "chunk_name": self.chunk_name,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "location": self.location,
            "context_summary": self.context_summary,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalFilters:
    """
    Filters for retrieval based on metadata.
    
    Leverages the rich metadata we've indexed.
    """
    
    repository_name: Optional[str] = None
    file_path: Optional[str] = None  # Exact or prefix match
    language: Optional[Language] = None
    chunk_type: Optional[str] = None  # function, class, method
    chunk_name_pattern: Optional[str] = None  # Regex pattern
    
    def to_chroma_filter(self) -> Optional[Dict[str, Any]]:
        """
        Convert to ChromaDB where filter.
        
        Returns:
            dict: ChromaDB filter or None if no filters
        """
        conditions = []

        if self.repository_name:
            conditions.append({"repository_name": self.repository_name})

        if self.file_path:
            if "*" in self.file_path or self.file_path.endswith("/"):
                conditions.append({
                    "file_path": {
                        "$contains": self.file_path.rstrip("/*")
                    }
                })
            else:
                conditions.append({"file_path": self.file_path})

        if self.language:
            language = self.language.value if hasattr(self.language, "value") else self.language
            conditions.append({"language": language})

        if self.chunk_type:
            conditions.append({"chunk_type": self.chunk_type})

        if self.chunk_name_pattern:
            conditions.append({
                "chunk_name": {
                    "$contains": self.chunk_name_pattern
                }
            })

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}
    
    def has_filters(self) -> bool:
        """Check if any filters are set."""
        return any([
            self.repository_name,
            self.file_path,
            self.language,
            self.chunk_type,
            self.chunk_name_pattern,
        ])


@dataclass
class RetrievalContext:
    """
    Context for a retrieval operation.
    
    Carries query, results, and metadata through the pipeline.
    """
    
    query: str
    results: List[RetrievalResult] = field(default_factory=list)
    filters: Optional[RetrievalFilters] = None
    
    # Statistics
    total_candidates: int = 0
    reranked_count: int = 0
    retrieval_time_seconds: float = 0.0
    
    def get_top_k(self, k: int = 5, score_field: str = "combined_score") -> List[RetrievalResult]:
        """
        Get top-k results by score.
        
        Args:
            k: Number of results
            score_field: Which score to use (combined_score, rerank_score, etc.)
            
        Returns:
            List[RetrievalResult]: Top-k results
        """
        # Sort by the specified score field
        sorted_results = sorted(
            self.results,
            key=lambda r: getattr(r, score_field) or 0.0,
            reverse=True
        )
        return sorted_results[:k]
    
    def filter_by_score(self, min_score: float, score_field: str = "combined_score") -> List[RetrievalResult]:
        """
        Filter results by minimum score.
        
        Args:
            min_score: Minimum score threshold
            score_field: Which score to use
            
        Returns:
            List[RetrievalResult]: Filtered results
        """
        return [
            r for r in self.results
            if (getattr(r, score_field) or 0.0) >= min_score
        ]
    
    def get_unique_files(self) -> List[str]:
        """Get list of unique file paths in results."""
        return list(set(r.file_path for r in self.results if r.file_path))
    
    def get_unique_repos(self) -> List[str]:
        """Get list of unique repositories in results."""
        return list(set(r.repository_name for r in self.results if r.repository_name))
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "total_results": len(self.results),
            "total_candidates": self.total_candidates,
            "reranked_count": self.reranked_count,
            "retrieval_time_seconds": self.retrieval_time_seconds,
            "unique_files": len(self.get_unique_files()),
            "unique_repos": len(self.get_unique_repos()),
            "results": [r.to_dict() for r in self.results],
        }
