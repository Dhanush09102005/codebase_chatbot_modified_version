"""
File scanner for finding and filtering code files in repositories.

Uses iterative traversal (os.walk) to handle deep directory structures,
and returns CodeFile objects enriched with language detection, file hashing,
and repository metadata.

Key features:
- Language enum for type-safe language detection
- SHA-256 file hashing for incremental indexing
- Repository name stored in CodeFile
- os.walk() with in-place dir filtering (production pattern)
"""
from pathlib import Path
from typing import List, Optional
import os
import hashlib

from app.core import logger, settings
from app.ingestion.models import CodeFile
from app.ingestion.language import Language


class FileScanner:
    """
    Scans repositories for supported code files while filtering out unwanted directories.
    
    Features:
    - Uses os.walk() with in-place filtering (production pattern)
    - Returns CodeFile objects with Language enum
    - Computes SHA-256 hash for incremental indexing
    - Stores repository name for multi-repo support
    - Configurable ignored directories and files
    - Detects hidden directories by default
    - Supports files without extensions (Dockerfile, Makefile, etc.)
    """
    
    def __init__(
        self,
        supported_extensions: Optional[List[str]] = None,
        ignored_dirs: Optional[List[str]] = None,
        ignored_files: Optional[List[str]] = None,
        allow_hidden: bool = False,
    ):
        """
        Initialize the file scanner with configurable settings.
        
        Args:
            supported_extensions: List of file extensions to scan. Defaults to settings.code_extensions
            ignored_dirs: Directories to skip. Defaults to settings.ignored_directories
            ignored_files: Files to skip. Defaults to settings.ignored_files
            allow_hidden: Whether to allow hidden directories (starting with .). Defaults to False
        """
        self.supported_extensions = set(
            e.lower() for e in (supported_extensions or settings.code_extensions)
        )
        self.ignored_dirs = set(ignored_dirs or settings.ignored_directories)
        self.ignored_files = set(ignored_files or settings.ignored_files)
        self.allow_hidden = allow_hidden
        
        logger.info(
            f"FileScanner initialized: "
            f"{len(self.supported_extensions)} extensions, "
            f"{len(self.ignored_dirs)} ignored dirs, "
            f"allow_hidden={self.allow_hidden}"
        )
    
    def scan(
    self,
    repo_path: Path,
    repository_name: Optional[str] = None,
) -> List[CodeFile]:
        """
        Scan repository and return CodeFile objects with language detection and hashing.
        
        Uses os.walk() with in-place directory filtering to avoid recursion limits.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            List[CodeFile]: List of CodeFile objects with metadata
            
        Example:
            >>> scanner = FileScanner()
            >>> files = scanner.scan(Path("/path/to/repo"))
            >>> for f in files:
            ...     print(f"{f.relative_path_str}: {f.language.value} ({f.content_hash[:8]})")
            src/main.py: python (a3f5e8d1)
            src/utils.py: python (b4d6c9a2)
            tests/test_main.py: python (c5e7f0b3)
        """
        if not repo_path.exists() or not repo_path.is_dir():
            logger.error(f"Invalid repository path: {repo_path}")
            return []
        
        repo_path = repo_path.resolve()
        repo_name = repository_name or repo_path.name
        logger.info(f"Scanning repository: {repo_path} (name: {repo_name})")
        
        code_files = []
        skipped_count = 0
        
        # Use os.walk() for efficient iterative traversal
        for root, dirs, files in os.walk(repo_path, followlinks=False):
            root_path = Path(root)
            
            # Filter directories in-place (prevents traversal)
            # This is why os.walk() is better than Path.rglob() for this use case
            dirs[:] = self._filter_directories(dirs, root_path)
            
            # Process files in this directory
            for file_name in files:
                file_path = root_path / file_name
                
                # Skip ignored files
                if file_name in self.ignored_files:
                    skipped_count += 1
                    continue
                
                # Check if file is supported
                if not self._is_supported_file(file_path):
                    skipped_count += 1
                    continue
                
                # Create CodeFile object with language detection and hashing
                try:
                    code_file = self._create_code_file(file_path, repo_path, repo_name)
                    code_files.append(code_file)
                except Exception as e:
                    logger.warning(f"Failed to create CodeFile for {file_path}: {e}")
                    skipped_count += 1
        
        logger.info(
            f"Scan complete: {len(code_files)} files found, {skipped_count} skipped in {repo_path}"
        )
        return code_files
    
    def _filter_directories(self, dirs: list, current_path: Path) -> list:
        """
        Filter directories in-place, removing those that should be ignored.
        
        This modifies the dirs list in-place so os.walk doesn't traverse them.
        This is the production pattern for efficient directory filtering.
        
        Args:
            dirs: List of directory names in current_path
            current_path: Current directory being walked
            
        Returns:
            list: Filtered list of directories to traverse
        """
        filtered = []
        
        for dir_name in dirs:
            # Skip ignored directories
            if dir_name in self.ignored_dirs:
                logger.debug(f"Skipping ignored directory: {dir_name}")
                continue
            
            # Skip hidden directories unless explicitly allowed
            if not self.allow_hidden and dir_name.startswith("."):
                logger.debug(f"Skipping hidden directory: {dir_name}")
                continue
            
            filtered.append(dir_name)
        
        return filtered
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """
        Check if file is supported (has correct extension or name).
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if file should be included
        """
        file_name = file_path.name
        
        # Check if filename has a known language mapping
        if Language.from_filename(file_name) != Language.UNKNOWN:
            return True
        
        # Check extension
        extension = file_path.suffix.lower()
        if extension in self.supported_extensions:
            return True
        
        return False
    
    def _detect_language(self, file_path: Path) -> Language:
        """
        Detect programming language from file path using Language enum.
        
        Uses multiple strategies:
        1. Exact filename match (Dockerfile → Language.DOCKERFILE)
        2. Extension match (main.py → Language.PYTHON)
        3. Default to Language.UNKNOWN
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language: Programming language enum
        """
        file_name = file_path.name
        
        # Strategy 1: Check exact filename
        lang = Language.from_filename(file_name)
        if lang != Language.UNKNOWN:
            return lang
        
        # Strategy 2: Check extension
        lang = Language.from_extension(file_path.suffix)
        if lang != Language.UNKNOWN:
            return lang
        
        # Default
        return Language.UNKNOWN
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA-256 hash of file content for incremental indexing.
        
        Reads file in chunks to handle large files efficiently.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: SHA-256 hash (hex string)
        """
        hash_obj = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                # Read in 64KB chunks for efficiency
                while chunk := f.read(65536):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute hash for {file_path}: {e}")
            # Return empty hash as fallback
            return ""
    
    def _create_code_file(
        self,
        file_path: Path,
        repo_root: Path,
        repo_name: str,
    ) -> CodeFile:
        """
        Create a CodeFile object from a file path.
        
        Args:
            file_path: Absolute path to the file
            repo_root: Root path of the repository
            repo_name: Name of the repository
            
        Returns:
            CodeFile: Enriched code file object with language, hash, and metadata
        """
        relative_path = file_path.relative_to(repo_root)
        size = file_path.stat().st_size
        language = self._detect_language(file_path)
        extension = file_path.suffix.lower() or "no_extension"
        content_hash = self._compute_file_hash(file_path)
        
        return CodeFile(
            absolute_path=file_path,
            relative_path=relative_path,
            repository_root=repo_root,
            repository_name=repo_name,
            language=language,
            extension=extension,
            size=size,
            content_hash=content_hash,
        )
    
    def get_file_stats(self, repo_path: Path) -> dict:
        """
        Get statistics about code files in repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            dict: Statistics including total files and breakdown by language/extension
        """
        files = self.scan(repo_path)
        
        # Count by language and extension
        language_counts = {}
        extension_counts = {}
        total_size = 0
        
        for code_file in files:
            lang_str = code_file.language.value
            language_counts[lang_str] = language_counts.get(lang_str, 0) + 1
            extension_counts[code_file.extension] = extension_counts.get(code_file.extension, 0) + 1
            total_size += code_file.size
        
        return {
            "total_files": len(files),
            "total_size_bytes": total_size,
            "by_language": language_counts,
            "by_extension": extension_counts,
            "repository": str(repo_path),
            "repository_name": repo_path.name,
        }
