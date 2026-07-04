"""
Language enumeration for code files.

Provides type-safe language detection and parser selection.
"""
from enum import Enum


class Language(str, Enum):
    """
    Supported programming languages.
    
    Using str Enum allows:
    - Type safety and autocomplete
    - Direct string comparison
    - Eliminates typos
    - Reliable parser selection
    """
    
    # Programming languages
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    CSHARP = "csharp"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    R = "r"
    OBJECTIVE_C = "objective-c"
    
    # Shell and scripting
    BASH = "bash"
    SHELL = "shell"
    POWERSHELL = "powershell"
    
    # Markup and data
    HTML = "html"
    CSS = "css"
    SCSS = "scss"
    SASS = "sass"
    YAML = "yaml"
    JSON = "json"
    XML = "xml"
    TOML = "toml"
    MARKDOWN = "markdown"
    SQL = "sql"
    
    # Build and config files
    DOCKERFILE = "dockerfile"
    MAKEFILE = "makefile"
    GROOVY = "groovy"
    CMAKE = "cmake"
    
    # Other
    GITIGNORE = "gitignore"
    DOCKERIGNORE = "dockerignore"
    REQUIREMENTS = "requirements"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, lang_str: str) -> "Language":
        """
        Convert string to Language enum, with fallback to UNKNOWN.
        
        Args:
            lang_str: Language as string (e.g., "python", "javascript")
            
        Returns:
            Language: Corresponding enum value or UNKNOWN
        """
        try:
            return cls(lang_str.lower())
        except (ValueError, AttributeError):
            return cls.UNKNOWN
    
    @classmethod
    def from_extension(cls, extension: str) -> "Language":
        """
        Map file extension to Language enum.
        
        Args:
            extension: File extension (e.g., ".py", ".js")
            
        Returns:
            Language: Corresponding language
        """
        mapping = {
            ".py": cls.PYTHON,
            ".js": cls.JAVASCRIPT,
            ".ts": cls.TYPESCRIPT,
            ".jsx": cls.JAVASCRIPT,
            ".tsx": cls.TYPESCRIPT,
            ".java": cls.JAVA,
            ".cpp": cls.CPP,
            ".cc": cls.CPP,
            ".cxx": cls.CPP,
            ".c": cls.C,
            ".h": cls.C,
            ".hpp": cls.CPP,
            ".go": cls.GO,
            ".rs": cls.RUST,
            ".rb": cls.RUBY,
            ".php": cls.PHP,
            ".cs": cls.CSHARP,
            ".swift": cls.SWIFT,
            ".kt": cls.KOTLIN,
            ".kts": cls.KOTLIN,
            ".scala": cls.SCALA,
            ".r": cls.R,
            ".m": cls.OBJECTIVE_C,
            ".sh": cls.BASH,
            ".bash": cls.BASH,
            ".zsh": cls.SHELL,
            ".fish": cls.SHELL,
            ".ps1": cls.POWERSHELL,
            ".html": cls.HTML,
            ".htm": cls.HTML,
            ".css": cls.CSS,
            ".scss": cls.SCSS,
            ".sass": cls.SASS,
            ".yaml": cls.YAML,
            ".yml": cls.YAML,
            ".json": cls.JSON,
            ".xml": cls.XML,
            ".sql": cls.SQL,
            ".md": cls.MARKDOWN,
            ".toml": cls.TOML,
            ".cmake": cls.CMAKE,
        }
        return mapping.get(extension.lower(), cls.UNKNOWN)
    
    @classmethod
    def from_filename(cls, filename: str) -> "Language":
        """
        Map specific filename to Language enum (files without extensions).
        
        Args:
            filename: Filename (e.g., "Dockerfile", "Makefile")
            
        Returns:
            Language: Corresponding language
        """
        mapping = {
            "dockerfile": cls.DOCKERFILE,
            "makefile": cls.MAKEFILE,
            "jenkinsfile": cls.GROOVY,
            "procfile": cls.BASH,
            ".gitignore": cls.GITIGNORE,
            ".dockerignore": cls.DOCKERIGNORE,
            "requirements.txt": cls.REQUIREMENTS,
            "package.json": cls.JSON,
            "pyproject.toml": cls.TOML,
            "cargo.toml": cls.TOML,
            "go.mod": cls.GO,
            "go.sum": cls.GO,
            "cmakelists.txt": cls.CMAKE,
        }
        return mapping.get(filename.lower(), cls.UNKNOWN)
    
    def __str__(self) -> str:
        """Return string value for display."""
        return self.value
