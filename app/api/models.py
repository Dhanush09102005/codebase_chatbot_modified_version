"""
API request and response models.

Pydantic models for API validation and documentation.

Includes input sanitization validators for:
- Repository URLs (must be valid Git URLs)
- Session IDs (must be valid UUIDs)
- Question length (prevent excessive prompts)
- Repository/file paths (prevent path traversal)
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
import re


# ==============================================================================
# Chat Sessions
# ==============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create a new chat session."""
    title: Optional[str] = Field(
        None,
        description="Optional session title. Auto-generated from first message if omitted."
    )


class SessionSummary(BaseModel):
    """Summary of a chat session (no messages)."""
    session_id: str = Field(..., description="Unique session ID")
    title: str = Field(..., description="Session title")
    created_at: str = Field(..., description="ISO timestamp of creation")
    updated_at: str = Field(..., description="ISO timestamp of last message")


class MessageInfo(BaseModel):
    """A single message in a session."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")
    timestamp: str = Field(..., description="ISO timestamp")


class SessionDetail(BaseModel):
    """Full session including all messages."""
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[MessageInfo] = Field(default_factory=list)


class ListSessionsResponse(BaseModel):
    """Response listing all sessions."""
    sessions: List[SessionSummary] = Field(default_factory=list)
    total_count: int


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

    @field_validator("repository_url")
    @classmethod
    def validate_git_url(cls, v: str) -> str:
        """Validate that repository_url is a valid Git URL."""
        if not v:
            raise ValueError("repository_url cannot be empty")

        # Check for valid Git URL patterns
        valid_patterns = [
            r"^https?://.*\.git$",                # https://github.com/user/repo.git
            r"^git@[\w\.-]+:[\w\-/]+\.git$",      # git@github.com:user/repo.git
            r"^git://",                            # git://...
            r"^https?://(github|gitlab|bitbucket)\.",  # Common platforms
        ]

        if not any(re.match(pattern, v) for pattern in valid_patterns):
            raise ValueError(
                "Invalid Git URL. Must be a HTTPS/SSH Git URL "
                "(e.g., https://github.com/user/repo.git)"
            )

        # Prevent local file paths disguised as URLs
        if v.startswith("file://") or "../" in v or "..\\" in v:
            raise ValueError("Local file URLs and path traversal are not allowed")

        return v

    @field_validator("branch")
    @classmethod
    def validate_branch(cls, v: Optional[str]) -> Optional[str]:
        """Validate branch name to prevent command injection."""
        if v is None:
            return v

        # Branch names must be safe (alphanumeric, dash, underscore, slash)
        if not re.match(r"^[\w\-/\.]+$", v):
            raise ValueError(
                "Invalid branch name. Only alphanumeric, dash, underscore, "
                "dot, and slash are allowed."
            )

        # Prevent command injection patterns
        dangerous = [";", "|", "&", "$", "`", "\n", "\r"]
        if any(char in v for char in dangerous):
            raise ValueError("Branch name contains invalid characters")

        return v


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
        examples=["How does FastAPI handle dependencies?"],
        min_length=3,
        max_length=2000,  # Prevent excessive prompts
    )
    session_id: Optional[str] = Field(
        None,
        description="Chat session ID for conversation memory. If provided, previous "
                    "messages in this session are included as context."
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
        description="Maximum tokens for LLM generation",
        ge=1,
        le=4096,  # Reasonable upper limit
    )
    temperature: Optional[float] = Field(
        0.1,
        description="LLM sampling temperature (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        """Sanitize question to prevent prompt injection."""
        # Strip excessive whitespace
        v = " ".join(v.split())

        # Check for suspiciously long repetitions (possible denial of service)
        if len(v) > 50:
            words = v.split()
            if len(set(words)) < len(words) * 0.3:  # >70% repetition
                raise ValueError("Question contains excessive repetition")

        return v

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate session_id is a valid UUID."""
        if v is None:
            return v

        # Must be a valid UUID format
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        if not re.match(uuid_pattern, v.lower()):
            raise ValueError("Invalid session ID format. Must be a valid UUID.")

        return v

    @field_validator("repository_name")
    @classmethod
    def sanitize_repository_name(cls, v: Optional[str]) -> Optional[str]:
        """Prevent path traversal in repository_name filter."""
        if v is None:
            return v

        # Block path traversal attempts
        if "../" in v or "..\\" in v or v.startswith("/") or v.startswith("\\"):
            raise ValueError("Invalid repository name: path traversal detected")

        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        """Validate language is alphanumeric."""
        if v is None:
            return v

        if not re.match(r"^[\w\-]+$", v):
            raise ValueError("Invalid language identifier")

        return v.lower()


class SourceReference(BaseModel):
    """Code source reference in the answer."""
    
    repository_name: str = Field(..., description="Repository name")
    file_path: str = Field(..., description="File path")
    chunk_type: str = Field(default="unknown", description="Chunk type (function, class, etc.)")
    chunk_name: str = Field(default="unknown", description="Function/class name")
    line_start: int = Field(default=0, description="Start line number")
    line_end: int = Field(default=0, description="End line number")
    relevance_score: float = Field(default=0.0, description="Relevance score")


class ChatResponse(BaseModel):
    """Response containing answer and sources."""
    
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    session_id: Optional[str] = Field(None, description="Session ID if memory was used")
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
