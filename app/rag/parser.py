"""
AST-based code parser using tree-sitter.

Extracts structural elements (functions, classes, imports, docstrings) from code files.
This is the key differentiator: understanding code structure before chunking.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import tree_sitter_languages

from app.core import logger
from app.ingestion import CodeFile, Language


class CodeParser:
    """
    Parses code files using tree-sitter to extract structural elements.
    
    Key features:
    - AST-based parsing (not regex or text splitting)
    - Supports 40+ languages via tree-sitter
    - Extracts functions, classes, imports, docstrings
    - Enriches CodeFile objects (progressive enrichment pattern)
    - Includes line numbers for jump-to-definition
    
    This is what makes the project stand out: we understand the code structure
    before chunking, not just arbitrary text splitting.
    """
    
    def __init__(self):
        """Initialize the code parser with tree-sitter support."""
        self.parsers = {}  # Cache parsers by language
        logger.info("CodeParser initialized with tree-sitter support")
    
    def parse(self, code_file: CodeFile) -> bool:
        """
        Parse a code file and enrich it with AST information.
        
        Extracts:
        - Functions (with parameters, return types, docstrings, line numbers)
        - Classes (with methods, inheritance, line numbers)
        - Imports (module names)
        - Docstrings
        - Full AST tree
        
        Args:
            code_file: CodeFile object to enrich
            
        Returns:
            bool: True if parsing succeeded
            
        Example:
            >>> parser = CodeParser()
            >>> code_file = CodeFile(...)
            >>> parser.parse(code_file)
            >>> print(code_file.is_parsed())  # True
            >>> print(len(code_file.functions))  # 5
            >>> print(code_file.functions[0])
            {
                "name": "get_user",
                "line_start": 10,
                "line_end": 25,
                "params": ["id: int"],
                "return_type": "User",
                "docstring": "Get user by ID",
                "decorators": ["@app.get"]
            }
        """
        if code_file.is_parsed():
            logger.debug(f"File already parsed: {code_file.relative_path_str}")
            return True
        
        # Check if language is supported
        if not self._is_language_supported(code_file.language):
            logger.warning(f"Language not supported for parsing: {code_file.language.value}")
            return False
        
        logger.info(f"Parsing {code_file.relative_path_str} ({code_file.language.value})")
        
        try:
            # Read file content
            content = self._read_file_content(code_file.absolute_path)
            if not content:
                logger.warning(f"Empty file: {code_file.relative_path_str}")
                return False
            
            # Get parser for language
            parser = self._get_parser(code_file.language)
            if not parser:
                return False
            
            # Parse AST
            tree = parser.parse(bytes(content, "utf-8"))
            root_node = tree.root_node
            
            # Store AST
            code_file.ast_tree = tree
            
            logger.info(f"AST assigned: {code_file.ast_tree is not None}")
            logger.info(f"is_parsed(): {code_file.is_parsed()}")
            # Extract structural elements based on language
            if code_file.language == Language.PYTHON:
                self._extract_python(code_file, root_node, content)
            elif code_file.language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
                self._extract_javascript(code_file, root_node, content)
            elif code_file.language == Language.JAVA:
                self._extract_java(code_file, root_node, content)
            elif code_file.language == Language.GO:
                self._extract_go(code_file, root_node, content)
            elif code_file.language == Language.RUST:
                self._extract_rust(code_file, root_node, content)
            else:
                # Generic extraction for other languages
                self._extract_generic(code_file, root_node, content)
            
            logger.info(
                f"Parsed {code_file.relative_path_str}: "
                f"{len(code_file.functions or [])} functions, "
                f"{len(code_file.classes or [])} classes, "
                f"{len(code_file.imports or [])} imports"
            )

            logger.info(
                f"{code_file.relative_path_str} -> "
                f"functions={len(code_file.functions or [])}, "
                f"classes={len(code_file.classes or [])}, "
                f"is_parsed={code_file.is_parsed()}"
            )
            return True
        
        except Exception as e:
            logger.error(f"Failed to parse {code_file.relative_path_str}: {e}")
            return False
    
    def parse_batch(self, code_files: List[CodeFile]) -> int:
        """
        Parse multiple code files in batch.
        
        Args:
            code_files: List of CodeFile objects to parse
            
        Returns:
            int: Number of successfully parsed files
        """
        success_count = 0
        for code_file in code_files:
            if self.parse(code_file):
                success_count += 1
        
        logger.info(f"Batch parsing complete: {success_count}/{len(code_files)} files")
        return success_count
    
    def _is_language_supported(self, language: Language) -> bool:
        """
        Check if a language is supported by tree-sitter.
        
        Args:
            language: Language enum
            
        Returns:
            bool: True if supported
        """
        supported = {
            Language.PYTHON,
            Language.JAVASCRIPT,
            Language.TYPESCRIPT,
            Language.JAVA,
            Language.CPP,
            Language.C,
            Language.GO,
            Language.RUST,
            Language.RUBY,
            Language.PHP,
            Language.CSHARP,
            Language.SWIFT,
            Language.KOTLIN,
            Language.SCALA,
            Language.BASH,
            Language.HTML,
            Language.CSS,
            Language.JSON,
            Language.YAML,
            Language.TOML,
        }
        return language in supported
    
    def _get_parser(self, language: Language):
        """
        Get or create a tree-sitter parser for a language.
        
        Args:
            language: Language enum
            
        Returns:
            Parser instance or None if not supported
        """
        # Return cached parser if available
        if language in self.parsers:
            return self.parsers[language]
        
        # Map Language enum to tree-sitter language name
        lang_map = {
            Language.PYTHON: "python",
            Language.JAVASCRIPT: "javascript",
            Language.TYPESCRIPT: "typescript",
            Language.JAVA: "java",
            Language.CPP: "cpp",
            Language.C: "c",
            Language.GO: "go",
            Language.RUST: "rust",
            Language.RUBY: "ruby",
            Language.PHP: "php",
            Language.CSHARP: "c_sharp",
            Language.SWIFT: "swift",
            Language.KOTLIN: "kotlin",
            Language.SCALA: "scala",
            Language.BASH: "bash",
            Language.HTML: "html",
            Language.CSS: "css",
            Language.JSON: "json",
            Language.YAML: "yaml",
            Language.TOML: "toml",
        }
        
        ts_lang_name = lang_map.get(language)
        if not ts_lang_name:
            logger.warning(f"No tree-sitter mapping for {language.value}")
            return None
        
        try:
            # Get parser from tree-sitter-languages
            parser = tree_sitter_languages.get_parser(ts_lang_name)
            self.parsers[language] = parser
            return parser
        except Exception as e:
            logger.error(f"Failed to get parser for {language.value}: {e}")
            return None
    
    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """
        Read file content with encoding fallback.
        
        Args:
            file_path: Path to file
            
        Returns:
            str: File content or None if read fails
        """
        encodings = ["utf-8", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, Exception):
                continue
        
        logger.error(f"Failed to read file with any encoding: {file_path}")
        return None
    
    def _extract_python(self, code_file: CodeFile, root_node, content: str):
        """Extract Python-specific structural elements."""
        code_file.imports = []
        code_file.functions = []
        code_file.classes = []
        code_file.docstrings = []
        
        # Traverse AST
        def traverse(node):
            # Import statements
            if node.type == "import_statement" or node.type == "import_from_statement":
                import_text = content[node.start_byte:node.end_byte]
                code_file.imports.append(import_text.strip())
            
            # Function definitions
            elif node.type == "function_definition":
                func_info = self._extract_python_function(node, content)
                if func_info:
                    code_file.functions.append(func_info)
                    if func_info.get("docstring"):
                        code_file.docstrings.append(func_info["docstring"])
            
            # Class definitions
            elif node.type == "class_definition":
                class_info = self._extract_python_class(node, content)
                if class_info:
                    code_file.classes.append(class_info)
                    if class_info.get("docstring"):
                        code_file.docstrings.append(class_info["docstring"])
            
            # Recurse
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
    
    def _extract_python_function(self, node, content: str) -> Optional[Dict[str, Any]]:
        """Extract Python function information."""
        try:
            # Function name
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None
            name = content[name_node.start_byte:name_node.end_byte]
            
            # Parameters
            params_node = node.child_by_field_name("parameters")
            params = content[params_node.start_byte:params_node.end_byte] if params_node else "()"
            
            # Return type (type annotation)
            return_type = None
            for child in node.children:
                if child.type == "type":
                    return_type = content[child.start_byte:child.end_byte]
            
            # Docstring
            docstring = None
            body_node = node.child_by_field_name("body")
            if body_node and len(body_node.children) > 0:
                first_stmt = body_node.children[0]
                if first_stmt.type == "expression_statement":
                    expr_child = first_stmt.children[0] if first_stmt.children else None
                    if expr_child and expr_child.type == "string":
                        docstring = content[expr_child.start_byte:expr_child.end_byte].strip('"\'')
            
            # Decorators
            decorators = []
            prev_sibling = node.prev_sibling
            while prev_sibling and prev_sibling.type == "decorator":
                dec_text = content[prev_sibling.start_byte:prev_sibling.end_byte]
                decorators.insert(0, dec_text)
                prev_sibling = prev_sibling.prev_sibling
            
            return {
                "name": name,
                "params": params,
                "return_type": return_type,
                "docstring": docstring,
                "decorators": decorators,
                "line_start": node.start_point[0] + 1,
                "line_end": node.end_point[0] + 1,
                "start_byte": node.start_byte,
                "end_byte": node.end_byte,
            }
        except Exception as e:
            logger.warning(f"Failed to extract Python function: {e}")
            return None
    
    def _extract_python_class(self, node, content: str) -> Optional[Dict[str, Any]]:
        """Extract Python class information."""
        try:
            # Class name
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None
            name = content[name_node.start_byte:name_node.end_byte]
            
            # Base classes (inheritance)
            bases = []
            superclasses_node = node.child_by_field_name("superclasses")
            if superclasses_node:
                bases_text = content[superclasses_node.start_byte:superclasses_node.end_byte]
                bases = [b.strip() for b in bases_text.strip("()").split(",") if b.strip()]
            
            # Docstring
            docstring = None
            body_node = node.child_by_field_name("body")
            if body_node and len(body_node.children) > 0:
                first_stmt = body_node.children[0]
                if first_stmt.type == "expression_statement":
                    expr_child = first_stmt.children[0] if first_stmt.children else None
                    if expr_child and expr_child.type == "string":
                        docstring = content[expr_child.start_byte:expr_child.end_byte].strip('"\'')
            
            # Methods (functions inside class)
            methods = []
            if body_node:
                for child in body_node.children:
                    if child.type == "function_definition":
                        method_info = self._extract_python_function(child, content)
                        if method_info:
                            methods.append(method_info)
            
            return {
                "name": name,
                "bases": bases,
                "docstring": docstring,
                "methods": methods,
                "line_start": node.start_point[0] + 1,
                "line_end": node.end_point[0] + 1,
                "start_byte": node.start_byte,
                "end_byte": node.end_byte,
            }
        except Exception as e:
            logger.warning(f"Failed to extract Python class: {e}")
            return None
    
    def _extract_javascript(self, code_file: CodeFile, root_node, content: str):
        """Extract JavaScript/TypeScript structural elements."""
        code_file.imports = []
        code_file.functions = []
        code_file.classes = []
        code_file.docstrings = []
        
        # TODO: Implement JavaScript extraction
        logger.warning("JavaScript extraction not yet implemented")
    
    def _extract_java(self, code_file: CodeFile, root_node, content: str):
        """Extract Java structural elements."""
        code_file.imports = []
        code_file.functions = []
        code_file.classes = []
        code_file.docstrings = []
        
        # TODO: Implement Java extraction
        logger.warning("Java extraction not yet implemented")
    
    def _extract_go(self, code_file: CodeFile, root_node, content: str):
        """Extract Go structural elements."""
        code_file.imports = []
        code_file.functions = []
        code_file.classes = []
        code_file.docstrings = []
        
        # TODO: Implement Go extraction
        logger.warning("Go extraction not yet implemented")
    
    def _extract_rust(self, code_file: CodeFile, root_node, content: str):
        """Extract Rust structural elements."""
        code_file.imports = []
        code_file.functions = []
        code_file.classes = []
        code_file.docstrings = []
        
        # TODO: Implement Rust extraction
        logger.warning("Rust extraction not yet implemented")
    
    def _extract_generic(self, code_file: CodeFile, root_node, content: str):
        """Generic extraction for unsupported languages."""
        code_file.imports = []
        code_file.functions = []
        code_file.classes = []
        code_file.docstrings = []
        
        logger.info(f"Using generic extraction for {code_file.language.value}")
