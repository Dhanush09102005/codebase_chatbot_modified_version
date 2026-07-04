"""
Repository manager for cloning and updating Git repositories.

Supports:
- GitHub, GitLab, Bitbucket, and any Git repository URL
- Private repositories via Git credential helpers (no token in URL)
- Repository caching with metadata manifest
- Branch selection during cloning
- Safe refresh with cleanliness verification
- SHA-256 based repository identifiers
"""
import inspect
import git
import shutil
import hashlib
import os
from pathlib import Path
from typing import Union, Optional
from urllib.parse import urlparse
from datetime import datetime
import shutil
import stat
from app.core import logger, settings
from app.ingestion.repository_metadata import RepositoryMetadata


def _remove_readonly(func, path, _):
     """Clear the readonly bit and retry deletion."""
     os.chmod(path, stat.S_IWRITE)
     func(path)

class RepositoryManager:
    """
    Manages repository cloning, caching, and updates.
    
    Features:
    - Supports any Git repository (GitHub, GitLab, Bitbucket, etc.)
    - Private repository support via Git credential helpers (secure)
    - Repository caching with metadata manifest
    - Branch selection support
    - Safe refresh with cleanliness verification
    - SHA-256 based identifiers
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        auth_token: Optional[str] = None,
    ):
        """
        Initialize the repository manager.
        
        Args:
            storage_path: Base directory for storing repos. 
                         Defaults to settings.repo_storage_path.
            auth_token: Git authentication token for private repos.
                       Defaults to settings.github_token.
                       Uses environment-based auth (not embedded in URL).
        """
        self.storage_path = storage_path or settings.repo_storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.auth_token = auth_token or settings.github_token
        self.hash_algorithm = settings.git_hash_algorithm or "sha256"
        
        logger.info(
            f"RepositoryManager initialized with storage: {self.storage_path}, "
            f"auth_enabled={'yes' if self.auth_token else 'no'}"
        )
    
    def load(
        self,
        source: Union[str, Path],
        refresh: bool = False,
        branch: Optional[str] = None,
    ) -> Path:
        """
        Load a repository from a Git URL or local path.
        
        Args:
            source: Git repository URL or local file path
            refresh: If True, update cached repo with git pull or reclone
            branch: Branch name to clone/checkout (e.g., 'main', 'develop')
            
        Returns:
            Path: Local path to the repository
            
        Raises:
            ValueError: If source is invalid
            git.GitCommandError: If cloning/pulling fails
            
        Example:
            >>> manager = RepositoryManager()
            >>> # Clone default branch
            >>> path = manager.load("https://github.com/user/repo.git")
            >>> # Clone specific branch
            >>> path = manager.load("https://github.com/user/repo.git", branch="develop")
            >>> # Refresh cached repo
            >>> path = manager.load("https://github.com/user/repo.git", refresh=True)
        """
        source_str = str(source)
        
        # Check if it's a Git URL
        if self._is_git_url(source_str):
            logger.info(f"Detected Git URL: {source_str}")
            logger.info(f"Storage path: {self.storage_path}")
            return self._clone_or_update_repository(
                source_str,
                refresh=refresh,
                branch=branch
            )
        
        # Check if it's a local path
        local_path = Path(source_str)
        if local_path.exists() and local_path.is_dir():
            logger.info(f"Using local repository: {local_path}")
            return local_path.resolve()
        
        raise ValueError(
            f"Invalid source: {source}. Must be a valid Git URL or existing local path."
        )
    
    def refresh(
        self,
        repo_path: Path,
        verify_clean: bool = True
    ) -> bool:
        """
        Update a cached repository with latest changes.
        
        Verifies repository cleanliness before pulling to prevent data loss.
        Attempts git pull first for efficiency. Falls back to full reclone if needed.
        
        Args:
            repo_path: Path to the cached repository
            verify_clean: If True, verify no local modifications before pulling
            
        Returns:
            bool: True if refresh succeeded, False otherwise
        """
        if not repo_path.exists():
            logger.error(f"Repository not found: {repo_path}")
            return False
        
        logger.info(f"Refreshing repository: {repo_path}")
        
        try:
            repo = git.Repo(repo_path)
            
            # Verify repository cleanliness
            if verify_clean and repo.is_dirty(untracked_files=True):
                logger.warning(
                    f"Repository has local modifications: {repo_path}. "
                    "Skipping refresh to prevent data loss."
                )
                return False
            
            # Try to pull latest changes
            try:
                origin = repo.remotes.origin
                origin.pull()
                
                # Update metadata
                metadata = RepositoryMetadata.from_file(repo_path)
                if metadata:
                    commit_sha = repo.head.commit.hexsha
                    metadata.update_refresh_time(commit_sha)
                    metadata.save(repo_path)
                
                logger.info(f"Successfully pulled latest changes for {repo_path}")
                return True
            
            except git.GitCommandError as e:
                logger.warning(f"Pull failed, attempting reclone: {e}")
                
                # Load metadata before deleting
                metadata = RepositoryMetadata.from_file(repo_path)
                
                # Fall back to reclone
                origin_url = repo.remotes.origin.url
                shutil.rmtree(repo_path)
                
                # Reclone with same branch if metadata exists
                branch = metadata.branch if metadata else None
                self._clone_repository(
                    origin_url,
                    repo_path,
                    branch=branch
                )
                
                logger.info(f"Successfully recloned repository to {repo_path}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to refresh repository: {e}")
            return False
    
    def _is_git_url(self, url: str) -> bool:
        """
        Check if the string is a valid Git repository URL.
        
        Supports:
        - GitHub: https://github.com/user/repo.git
        - GitLab: https://gitlab.com/user/repo.git
        - Bitbucket: https://bitbucket.org/user/repo.git
        - Generic: any URL with git protocol or ending in .git
        
        Args:
            url: String to check
            
        Returns:
            bool: True if it's a valid Git URL
        """
        try:
            # Check for .git protocol
            if url.startswith("git@"):
                return True
            
            # Check for git:// protocol
            if url.startswith("git://"):
                return True
            
            # Check for http(s):// protocol with .git extension
            if url.endswith(".git") and ("http://" in url or "https://" in url):
                return True
            
            # Parse URL and check if it looks like a repo
            parsed = urlparse(url)
            if parsed.scheme in ("http", "https"):
                # Common git hosting platforms
                git_platforms = {
                    "github.com", "gitlab.com", "bitbucket.org",
                    "gitea", "gitee.com", "codeberg.org"
                }
                return any(platform in parsed.netloc for platform in git_platforms)
            
            return False
        except Exception:
            return False
    
    def _clone_or_update_repository(
        self,
        git_url: str,
        refresh: bool = False,
        branch: Optional[str] = None,
    ) -> Path:
        """
        Clone a new repository or update existing cached one.
        
        Args:
            git_url: Git repository URL
            refresh: If True, update cached repo instead of reusing
            branch: Branch name to clone/checkout
            
        Returns:
            Path: Local path to repository
            
        Raises:
            git.GitCommandError: If cloning/pulling fails
        """
        repo_name = self._extract_repo_name(git_url)
        repo_hash = self._hash_url(git_url)
        target_dir = self.storage_path / f"{repo_name}_{repo_hash}"
        
        # If already exists
        if target_dir.exists():
            if refresh:
                logger.info(f"Refreshing existing repository: {target_dir}")
                if self.refresh(target_dir):
                    return target_dir
                else:
                    # If refresh failed, try reclone
                    logger.info("Refresh failed, attempting fresh clone")
                    shutil.rmtree(target_dir)
            else:
                logger.info(f"Using cached repository: {target_dir}")
                return target_dir
        
        # Clone the repository
        self._clone_repository(git_url, target_dir, branch=branch)
        return target_dir
    
    def _clone_repository(
        self,
        git_url: str,
        target_dir: Path,
        branch: Optional[str] = None,
    ) -> None:
        """
        Clone a repository using Git credential helpers (secure auth).
        
        Uses environment variables for authentication instead of embedding
        tokens in the URL, preventing accidental token exposure in logs.
        
        Args:
            git_url: Git repository URL
            target_dir: Target directory for cloning
            branch: Branch name to clone
            
        Raises:
            git.GitCommandError: If cloning fails
        """
        logger.info(f"Cloning {git_url} to {target_dir}" + (f" (branch: {branch})" if branch else ""))
        
        # Ensure target directory is absolute and resolved
        target_dir = target_dir.resolve()
        
        # Ensure parent directory exists
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup environment for authentication
        env = os.environ.copy()
        if self.auth_token:
            # Use Git credential helper environment variables
            # This keeps the token out of the URL and logs
            env['GIT_ASKPASS'] = 'echo'
            env['GIT_USERNAME'] = 'oauth'
            env['GIT_PASSWORD'] = self.auth_token
        
        try:
            # Clone with optional branch
            # Convert Path to string for GitPython compatibility
            clone_kwargs = {
                'depth': 1,  # Shallow clone
                'env': env if self.auth_token else None,
            }
            if branch:
                clone_kwargs['branch'] = branch
            
            repo = git.Repo.clone_from(git_url, str(target_dir), **clone_kwargs)
            
            # Get current commit SHA
            commit_sha = repo.head.commit.hexsha
            
            # Save metadata
            metadata = RepositoryMetadata(
                url=git_url,
                branch=branch or repo.active_branch.name,
                clone_time=datetime.now().isoformat(),
                commit_sha=commit_sha,
                repository_name=self._extract_repo_name(git_url),
                repository_id=self._hash_url(git_url)
            )
            metadata.save(target_dir)
            
            logger.info(f"Successfully cloned repository to {target_dir}")
        
        except git.GitCommandError as e:
            logger.error(f"Failed to clone repository: {e}")
            # Clean up partial clone
            if target_dir.exists():
                shutil.rmtree(target_dir)
            raise
    
    def _hash_url(self, url: str) -> str:
        """
        Generate a unique hash for a repository URL.
        
        Uses SHA-256 for modern security practices.
        Returns first 12 characters for readable folder names.
        
        Args:
            url: Repository URL
            
        Returns:
            str: Hash string (first 12 chars of SHA-256)
        """
        hash_obj = hashlib.sha256(url.encode())
        return hash_obj.hexdigest()[:12]
    
    def _extract_repo_name(self, git_url: str) -> str:
        """
        Extract repository name from Git URL.
        
        Examples:
        - https://github.com/user/myrepo.git → myrepo
        - git@github.com:user/repo → repo
        - https://gitlab.com/org/group/project.git → project
        
        Args:
            git_url: Git repository URL
            
        Returns:
            str: Repository name (just the last component)
        """
        # Remove .git suffix if present
        url = git_url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        
        # Extract last path component
        if "://" in url:  # HTTP/HTTPS format
            # Split by / and get the last non-empty component
            parts = [p for p in url.split("/") if p]
            if parts:
                return parts[-1]
        elif ":" in url:  # SSH format (git@github.com:user/repo)
            path = url.split(":")[-1]
            parts = [p for p in path.split("/") if p]
            if parts:
                return parts[-1]
        
        # Fallback
        return "unknown_repo"
    
    
    


    def delete_repository(self, repo_path: Path) -> bool:
        """
        Delete a cached repository from storage.

        Args:
            repo_path: Path to the repository

        Returns:
            bool: True if deleted successfully
        """
        try:
            # Safety check
            if not repo_path.is_relative_to(self.storage_path):
                logger.warning(f"Cannot delete {repo_path} - not in storage path")
                return False
            print(inspect.getfile(RepositoryManager))
            print("_remove_readonly" in globals())
            print(globals().get("_remove_readonly"))
            shutil.rmtree(repo_path, onerror=_remove_readonly)

            logger.info(f"Deleted repository at {repo_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete repository: {e}")
            return False
    
    def get_cached_repositories(self) -> list[dict]:
        """
        List all cached repositories with metadata.
        
        Returns:
            list[dict]: List of cached repos with metadata
        """
        repos = []
        for repo_dir in self.storage_path.iterdir():
            if not repo_dir.is_dir():
                continue
            
            try:
                # Try to load metadata first
                metadata = RepositoryMetadata.from_file(repo_dir)
                if metadata:
                    repos.append({
                        "name": repo_dir.name,
                        "repository_id": self._hash_url(metadata.url),
                        "path": str(repo_dir),
                        "url": metadata.url,
                        "branch": metadata.branch,
                        "clone_time": metadata.clone_time,
                        "last_refresh": metadata.last_refresh_time,
                        "commit_sha": metadata.commit_sha,
                    })
                else:
                    # Fallback: read from Git repo
                    repo = git.Repo(repo_dir)
                    origin_url = repo.remotes.origin.url if repo.remotes else "unknown"
                    repos.append({
                        "name": repo_dir.name,
                        "path": str(repo_dir),
                        "url": origin_url,
                        "branch": None,
                        "clone_time": None,
                        "last_refresh": None,
                        "commit_sha": None,
                    })
            except Exception as e:
                logger.warning(f"Could not read repo info for {repo_dir}: {e}")
        
        return repos
    
    def get_repository_info(self, repo_path: Path) -> Optional[dict]:
        """
        Get detailed information about a cached repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            dict: Repository information or None if not found
        """
        if not repo_path.exists():
            return None
        
        try:
            repo = git.Repo(repo_path)
            metadata = RepositoryMetadata.from_file(repo_path)
            
            return {
                "path": str(repo_path),
                "url": metadata.url if metadata else repo.remotes.origin.url,
                "branch": metadata.branch if metadata else repo.active_branch.name,
                "commit_sha": repo.head.commit.hexsha,
                "is_dirty": repo.is_dirty(untracked_files=True),
                "clone_time": metadata.clone_time if metadata else None,
                "last_refresh": metadata.last_refresh_time if metadata else None,
            }
        except Exception as e:
            logger.error(f"Failed to get repository info: {e}")
            return None
