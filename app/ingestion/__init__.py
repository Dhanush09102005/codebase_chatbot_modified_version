"""Repository ingestion module for cloning and scanning code repositories."""
from app.ingestion.language import Language
from app.ingestion.models import CodeFile
from app.ingestion.repository_metadata import RepositoryMetadata
from app.ingestion.repository_manager import RepositoryManager
from app.ingestion.file_scanner import FileScanner

# Backward compatibility alias
RepositoryLoader = RepositoryManager

__all__ = [
    "Language",
    "CodeFile",
    "RepositoryMetadata",
    "RepositoryManager",
    "RepositoryLoader",
    "FileScanner",
]
