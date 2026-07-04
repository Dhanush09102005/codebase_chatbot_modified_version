"""
AST-based semantic chunker.

This is the key differentiator: chunks correspond to code structure
(functions, classes, modules) not arbitrary text splits.

Traditional RAG:
    Chunk 1: characters 0-512 (might split mid-function)
    Chunk 2: characters 513-1024 (arbitrary)

Our approach:
    Chunk 1: def create_access_token() - complete function
    Chunk 2: class UserService - complete class
    Chunk 3: import block
"""
from pathlib import Path
from typing import List, Optional

from app.core import logger, settings
from app.ingestion import CodeFile, Language
from app.rag.base_chunker import BaseChunker, TextChunker
from app.rag.chunk_models import CodeChunk, ChunkingResult


class ASTChunker(BaseChunker):
    """
    Chunks code based on AST structure (functions, classes, imports).
    
    Key differentiator: Semantic units, not arbitrary text splits.
    
    Each chunk is a complete:
    - Function (with decorators, docstring, body)
    - Class (with methods, inheritance)
    - Import block
    - Module-level code
    
    This preserves context and improves retrieval quality dramatically.
    """
    
    def __init__(
        self,
        include_imports: bool = True,
        include_docstrings: bool = True,
        max_chunk_size: int = 4000,
    ):
        """
        Initialize AST chunker.
        
        Args:
            include_imports: Whether to create chunks for import blocks
            include_docstrings: Whether to include docstrings in function/class chunks
            max_chunk_size: Max chunk size in characters (for splitting large classes)
        """
        self.include_imports = include_imports
        self.include_docstrings = include_docstrings
        self.max_chunk_size = max_chunk_size
        
        # Fallback chunker for unsupported languages
        self.text_chunker = TextChunker()
        
        logger.info(
            f"ASTChunker initialized: "
            f"imports={include_imports}, "
            f"docstrings={include_docstrings}, "
            f"max_size={max_chunk_size}"
        )
    
    def chunk(self, code_file: CodeFile) -> bool:
        """
        Chunk a code file based on its AST structure.
        
        Requires the file to be parsed first (code_file.is_parsed() == True).
        
        Args:
            code_file: CodeFile to chunk (will be enriched in-place)
            
        Returns:
            bool: True if chunking succeeded
        """
        if code_file.is_chunked():
            logger.debug(f"File already chunked: {code_file.relative_path_str}")
            return True
        
        # Check if file has been parsed
        if not code_file.is_parsed():
            logger.warning(
                f"File not parsed, cannot use AST chunking: {code_file.relative_path_str}"
            )
            # Fallback to text chunking
            return self.text_chunker.chunk(code_file)
        
        logger.info(f"AST-chunking {code_file.relative_path_str}")
        
        try:
            # Read file content
            content = self._read_file(code_file.absolute_path)
            if not content:
                logger.warning(f"Empty file: {code_file.relative_path_str}")
                return False
            
            chunks = []
            
            # Chunk imports
            if self.include_imports and code_file.imports:
                import_chunk = self._create_import_chunk(code_file, content)
                if import_chunk:
                    chunks.append(import_chunk)
            
            # Chunk functions
            if code_file.functions:
                for func in code_file.functions:
                    func_chunk = self._create_function_chunk(code_file, func, content)
                    if func_chunk:
                        chunks.append(func_chunk)
            
            # Chunk classes
            if code_file.classes:
                for cls in code_file.classes:
                    class_chunks = self._create_class_chunks(code_file, cls, content)
                    chunks.extend(class_chunks)
            
            # Enrich CodeFile (progressive enrichment pattern)
            code_file.chunks = chunks
            
            logger.info(
                f"AST-chunked {code_file.relative_path_str}: "
                f"{len(chunks)} chunks "
                f"({len(code_file.functions or [])} functions, "
                f"{len(code_file.classes or [])} classes)"
            )
            return True
        
        except Exception as e:
            logger.error(f"Failed to AST-chunk {code_file.relative_path_str}: {e}")
            return False
    
    def _create_import_chunk(
        self,
        code_file: CodeFile,
        content: str
    ) -> Optional[CodeChunk]:
        """
        Create a chunk for import statements.
        
        Groups all imports into a single chunk for context.
        """
        if not code_file.imports:
            return None
        
        import_text = "\n".join(code_file.imports)
        
        # Estimate line numbers
        line_start = 1
        line_end = import_text.count('\n') + 1
        
        return CodeChunk(
            text=import_text,
            chunk_type="import",
            name="imports",
            file_path=code_file.relative_path_str,
            repository_name=code_file.repository_name,
            language=code_file.language,
            line_start=line_start,
            line_end=line_end,
            metadata={
                "import_count": len(code_file.imports),
                "imports": code_file.imports,
            }
        )
    
    def _create_function_chunk(
        self,
        code_file: CodeFile,
        func_info: dict,
        content: str
    ) -> Optional[CodeChunk]:
        """
        Create a chunk for a function.
        
        This is a complete semantic unit: decorator + signature + docstring + body.
        """
        try:
            # Extract function text from content
            start_byte = func_info.get("start_byte", 0)
            end_byte = func_info.get("end_byte", len(content))
            func_text = content[start_byte:end_byte]
            
            # Build metadata
            metadata = {
                "params": func_info.get("params", "()"),
                "return_type": func_info.get("return_type"),
                "decorators": func_info.get("decorators", []),
            }
            
            # Add docstring to metadata
            if self.include_docstrings and func_info.get("docstring"):
                metadata["docstring"] = func_info["docstring"]
            
            return CodeChunk(
                text=func_text,
                chunk_type="function",
                name=func_info["name"],
                file_path=code_file.relative_path_str,
                repository_name=code_file.repository_name,
                language=code_file.language,
                line_start=func_info["line_start"],
                line_end=func_info["line_end"],
                metadata=metadata
            )
        
        except Exception as e:
            logger.warning(f"Failed to create function chunk: {e}")
            return None
    
    def _create_class_chunks(
        self,
        code_file: CodeFile,
        class_info: dict,
        content: str
    ) -> List[CodeChunk]:
        """
        Create chunk(s) for a class.
        
        If class is small, creates one chunk.
        If class is large (>max_chunk_size), splits into:
        - Class header + docstring
        - Individual methods
        """
        chunks = []
        
        try:
            # Extract class text
            start_byte = class_info.get("start_byte", 0)
            end_byte = class_info.get("end_byte", len(content))
            class_text = content[start_byte:end_byte]
            
            # Build metadata
            metadata = {
                "bases": class_info.get("bases", []),
                "method_count": len(class_info.get("methods", [])),
                "method_names": [m["name"] for m in class_info.get("methods", [])],
            }
            
            if self.include_docstrings and class_info.get("docstring"):
                metadata["docstring"] = class_info["docstring"]
            
            # If class is small, create single chunk
            if len(class_text) <= self.max_chunk_size:
                chunk = CodeChunk(
                    text=class_text,
                    chunk_type="class",
                    name=class_info["name"],
                    file_path=code_file.relative_path_str,
                    repository_name=code_file.repository_name,
                    language=code_file.language,
                    line_start=class_info["line_start"],
                    line_end=class_info["line_end"],
                    metadata=metadata
                )
                chunks.append(chunk)
            else:
                # Large class: chunk methods individually
                logger.debug(
                    f"Large class {class_info['name']} ({len(class_text)} chars), "
                    "chunking methods individually"
                )
                
                # Create header chunk (class definition + docstring)
                header_lines = class_info.get("line_start", 1)
                first_method_line = (
                    class_info["methods"][0]["line_start"]
                    if class_info.get("methods")
                    else class_info["line_end"]
                )
                
                header_chunk = CodeChunk(
                    text=class_text[:500],  # First 500 chars as header
                    chunk_type="class_header",
                    name=f"{class_info['name']}_header",
                    file_path=code_file.relative_path_str,
                    repository_name=code_file.repository_name,
                    language=code_file.language,
                    line_start=header_lines,
                    line_end=first_method_line - 1,
                    metadata=metadata
                )
                chunks.append(header_chunk)
                
                # Create chunks for each method
                for method_info in class_info.get("methods", []):
                    method_chunk = self._create_method_chunk(
                        code_file,
                        class_info["name"],
                        method_info,
                        content
                    )
                    if method_chunk:
                        chunks.append(method_chunk)
        
        except Exception as e:
            logger.warning(f"Failed to create class chunks: {e}")
        
        return chunks
    
    def _create_method_chunk(
        self,
        code_file: CodeFile,
        class_name: str,
        method_info: dict,
        content: str
    ) -> Optional[CodeChunk]:
        """
        Create a chunk for a class method.
        """
        try:
            # Extract method text
            start_byte = method_info.get("start_byte", 0)
            end_byte = method_info.get("end_byte", len(content))
            method_text = content[start_byte:end_byte]
            
            metadata = {
                "class_name": class_name,
                "params": method_info.get("params", "()"),
                "return_type": method_info.get("return_type"),
                "decorators": method_info.get("decorators", []),
            }
            
            if self.include_docstrings and method_info.get("docstring"):
                metadata["docstring"] = method_info["docstring"]
            
            return CodeChunk(
                text=method_text,
                chunk_type="method",
                name=f"{class_name}.{method_info['name']}",
                file_path=code_file.relative_path_str,
                repository_name=code_file.repository_name,
                language=code_file.language,
                line_start=method_info["line_start"],
                line_end=method_info["line_end"],
                metadata=metadata
            )
        
        except Exception as e:
            logger.warning(f"Failed to create method chunk: {e}")
            return None
    
    def _read_file(self, file_path: Path) -> str:
        """
        Read file content with encoding fallback.
        
        Args:
            file_path: Path to file
            
        Returns:
            str: File content or empty string
        """
        encodings = ["utf-8", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, Exception):
                continue
        
        logger.error(f"Failed to read file: {file_path}")
        return ""
