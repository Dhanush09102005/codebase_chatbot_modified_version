"""
Configuration management using Pydantic Settings.
Loads environment variables from .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    llm_provider: str = "huggingface"
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct" # Model name for provider
    llm_max_tokens: int = 512  # Maximum tokens for LLM generation
    llm_temperature: float = 0.1  # Sampling temperature (0.0-1.0)
    llm_device: str = "cpu"  # "cpu" or "cuda"
    
    # HuggingFace Configuration (for HuggingFace LLM)
    hf_token: Optional[str] = None
    
    # API Keys (for different LLM providers)
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    
    # Prompt Builder Configuration
    prompt_include_line_numbers: bool = True
    prompt_include_chunk_types: bool = True
    prompt_max_chunks: int = 5
    prompt_instruction_style: str = "strict"  # "strict" or "flexible"
    
    # Embedding Model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # ChromaDB Configuration
    chroma_persist_directory: Path = Path("./data/chroma_db")
    
    # Repository Storage
    repo_storage_path: Path = Path("./data/repos")
    
    # Repository Management
    github_token: Optional[str] = None  # GitHub Personal Access Token for private repos
    git_hash_algorithm: str = "sha256"  # Hash algorithm for repo identifiers (sha256, sha1)
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    log_directory: Path = Path("./logs")
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB per log file
    log_backup_count: int = 5  # Keep 5 backup files
    
    # RAG Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    hybrid_alpha: float = 0.5  # Balance between BM25 and vector similarity
    
    # Retrieval Configuration
    retrieval_top_k: int = 10  # Initial candidates
    rerank_top_k: int = 5  # Final results after reranking
    min_similarity_score: float = 0.3  # Minimum score threshold
    enable_metadata_filtering: bool = True  # Filter by language, repo, etc.
    enable_reranking: bool = True  # Rerank results by relevance
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_max_length: int = 512  # Max tokens for embedding model
    embedding_normalize: bool = True  # Normalize embeddings to unit vectors
    embedding_device: str = "cpu"  # "cpu" or "cuda"
    
    # ChromaDB Configuration
    chroma_persist_directory: Path = Path("./data/chroma_db")
    chroma_collection_name: str = "code_chunks"
    chroma_distance_metric: str = "cosine"  # "cosine", "l2", "ip"
    
    # Supported file extensions for code
    code_extensions: list[str] = [
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".cpp", ".c", ".h", ".hpp",
        ".go", ".rs", ".rb", ".php", ".cs",
        ".swift", ".kt", ".scala", ".r", ".m"
    ]
    
    # File scanner configuration
    # Directories to ignore during repository scanning
    ignored_directories: list[str] = [
        ".git", ".github", ".gitlab", ".gitignore",
        "node_modules", "venv", "env", ".venv", ".env",
        "__pycache__", ".pytest_cache", ".tox",
        "dist", "build", "target",
        ".idea", ".vscode", ".vs",
        "vendor", "deps", "_build",
        ".next", ".nuxt", ".turbo",
        "coverage", ".coverage", "htmlcov",
        "site-packages", "bower_components",
        ".terraform", ".cargo", "bin", "obj",
        "out", ".gradle", ".mvn",
        "tmp", "temp", ".tmp",
    ]
    
    # Files to ignore during repository scanning
    ignored_files: list[str] = [
        ".DS_Store", "Thumbs.db",
        "package-lock.json", "yarn.lock", "poetry.lock",
        "Cargo.lock", "Gemfile.lock", "go.sum",
        ".env", ".env.local", ".env.*.local",
    ]
    
    # Allow hidden directories (starting with .) by default
    allow_hidden_directories: bool = False
    
    # Model configuration (using Pydantic v2 style)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
        self.repo_storage_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
