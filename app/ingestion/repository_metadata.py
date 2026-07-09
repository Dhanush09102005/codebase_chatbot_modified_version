"""
Repository metadata management for cached repositories.

Stores metadata in a JSON manifest file for debugging and management.
"""
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class RepositoryMetadata:
    """
    Metadata about a cached repository.
    
    Stored in .repo_metadata.json in each cached repository.
    """
    url: str
    branch: Optional[str]
    clone_time: str
    last_refresh_time: Optional[str] = None
    commit_sha: Optional[str] = None
    repository_name: str = "unknown"
    repository_id: Optional[str] = None
    
    @classmethod
    def from_file(cls, repo_path: Path) -> Optional["RepositoryMetadata"]:
        """
        Load metadata from repository's metadata file.
        
        Args:
            repo_path: Path to the cached repository
            
        Returns:
            RepositoryMetadata or None if file doesn't exist
        """
        metadata_file = repo_path / ".repo_metadata.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**data)
        except Exception:
            return None
    
    def save(self, repo_path: Path) -> bool:
        """
        Save metadata to repository's metadata file.
        
        Args:
            repo_path: Path to the cached repository
            
        Returns:
            bool: True if saved successfully
        """
        metadata_file = repo_path / ".repo_metadata.json"
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
            return True
        except Exception:
            return False
    
    def update_refresh_time(self, commit_sha: Optional[str] = None) -> None:
        """
        Update the last refresh time and optionally the commit SHA.
        
        Args:
            commit_sha: Latest commit SHA after refresh
        """
        self.last_refresh_time = datetime.now().isoformat()
        if commit_sha:
            self.commit_sha = commit_sha
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
