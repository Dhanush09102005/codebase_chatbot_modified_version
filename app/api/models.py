"""
API request and response models.

Pydantic models for API validation and documentation.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ==============================================================================
# Repository Indexing
# ==============================================================================

class IndexRepositoryRequest(BaseModel):
    """Request to index a repository."""
    
    repository_url: str = Field(
        ...,
        description="Git repository URL (GitHub, GitLab, etc.)",
        examples=["https://github.com/fastapi/fastapi.git"]
    )
    branch: Optional[str] = Field(
        None,
        description="Git branch to index (default: main/master)"
    )
    force_refresh: bool = Field(
        False,
        description="Force re-indexing even if already cached"
    )


class IndexingProgress(BaseModel):
    """Progress information during indexing."""
    
    stage: str = Field(..., description="Current stage")
    files_processed: int = Field(0, description="Files processed so far")
    total_files: int = Field(0, description="Total files to process")
    chunks_indexed: int = Field(0, description="Chunks indexed so far")


class IndexRepositoryResponse(BaseModel):
    """Response from repository indexing."""
    
    repository_id: str = Field(..., description="Unique repository identifier")
    repository_name: str = Field(..., description="Repository name")
    status: str = Field(..., description="Indexing status")
    files_processed: int = Field(..., description="Number of files processed")
    chunks_indexed: int = Field(..., description="Number of chunks indexed")
    indexing_time_seconds: float = Field(..., description="Total indexing time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ==============================================================================
# Chat / Q&A
# ==============================================================================

class ChatRequest(BaseModel):
    """Request to answer a question about code."""
    
    question: str = Field(
        ...,
        description="Question about the code",
        examples=["How does FastAPI handle dependencies?"]
    )
    repository_id: Optional[str] = Field(
        None,
        description="Filter by specific repository ID"
    )
    repository_name: Optional[str] = Field(
        None,
        description="Filter by repository name"
    )
    language: Optional[str] = Field(
        None,
        description="Filter by programming language",
        examples=["python", "javascript"]
    )
    max_tokens: Optional[int] = Field(
        512,
        description="Maximum tokens for LLM generation"
    )
    temperature: Optional[float] = Field(
        0.1,
        description="LLM sampling temperature (0.0-1.0)",
        ge=0.0,
        le=1.0
    )


class SourceReference(BaseModel):
    """Code source reference in the answer."""
    
    repository_name: str = Field(..., description="Repository name")
    file_path: str = Field(..., description="File path")
    chunk_type: str = Field(..., description="Chunk type (function, class, etc.)")
    chunk_name: str = Field(..., description="Function/class name")
    line_start: int = Field(..., description="Start line number")
    line_end: int = Field(..., description="End line number")
    relevance_score: float = Field(..., description="Relevance score")


class ChatResponse(BaseModel):
    """Response containing answer and sources."""
    
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Code sources used in answer"
    )
    retrieval_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Retrieval statistics"
    )
    llm_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="LLM generation statistics"
    )
    total_time_seconds: float = Field(..., description="Total query time")


# ==============================================================================
# Repository Management
# ==============================================================================

class RepositoryInfo(BaseModel):
    """Information about an indexed repository."""
    
    repository_id: str = Field(..., description="Unique repository identifier")
    repository_name: str = Field(..., description="Repository name")
    repository_url: str = Field(..., description="Git repository URL")
    indexed_at: str = Field(..., description="Indexing timestamp")
    files_count: int = Field(..., description="Number of indexed files")
    chunks_count: int = Field(..., description="Number of indexed chunks")
    languages: List[str] = Field(
        default_factory=list,
        description="Programming languages in repository"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class ListRepositoriesResponse(BaseModel):
    """Response listing all indexed repositories."""
    
    repositories: List[RepositoryInfo] = Field(
        default_factory=list,
        description="List of indexed repositories"
    )
    total_count: int = Field(..., description="Total number of repositories")
    total_chunks: int = Field(..., description="Total chunks across all repositories")


class DeleteRepositoryResponse(BaseModel):
    """Response from deleting a repository."""
    
    repository_id: str = Field(..., description="Deleted repository ID")
    status: str = Field(..., description="Deletion status")
    chunks_deleted: int = Field(..., description="Number of chunks deleted")


# ==============================================================================
# Health Check
# ==============================================================================

class HealthCheck(BaseModel):
    """System health check response."""
    
    status: str = Field(..., description="Overall system status")
    vector_store: Dict[str, Any] = Field(
        default_factory=dict,
        description="Vector store health"
    )
    embedding_model: Dict[str, Any] = Field(
        default_factory=dict,
        description="Embedding model health"
    )
    llm_provider: Dict[str, Any] = Field(
        default_factory=dict,
        description="LLM provider health"
    )
    timestamp: str = Field(..., description="Health check timestamp")
