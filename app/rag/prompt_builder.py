"""
Prompt builder for RAG queries.

The prompt builder ONLY formats prompts.
It does NOT retrieve chunks, call LLMs, or orchestrate workflows.
"""
from typing import List, Optional
from app.rag.retrieval_models import RetrievalResult


class PromptBuilder:
    """
    Builds structured prompts for code Q&A.
    
    Key features:
    - Formats retrieved code chunks into structured prompts
    - Includes repository names, file paths, line numbers
    - Instructs model to answer only from context
    - Instructs model to admit when answer not found
    - Customizable templates
    
    This is ONLY a prompt formatter - no retrieval, no LLM calls.
    """
    
    def __init__(
        self,
        include_line_numbers: bool = True,
        include_chunk_types: bool = True,
        max_chunks_in_prompt: int = 5,
        instruction_style: str = "strict"
    ):
        """
        Initialize prompt builder.
        
        Args:
            include_line_numbers: Include line numbers in code chunks
            include_chunk_types: Include chunk type (function, class, etc.)
            max_chunks_in_prompt: Maximum chunks to include
            instruction_style: "strict" or "flexible"
        """
        self.include_line_numbers = include_line_numbers
        self.include_chunk_types = include_chunk_types
        self.max_chunks_in_prompt = max_chunks_in_prompt
        self.instruction_style = instruction_style
    
    def build_prompt(
        self,
        question: str,
        retrieval_results: List[RetrievalResult],
        custom_instruction: Optional[str] = None
    ) -> str:
        """
        Build prompt from question and retrieval results.
        
        Args:
            question: User's question
            retrieval_results: Retrieved code chunks
            custom_instruction: Optional custom instruction
            
        Returns:
            str: Formatted prompt
        """
        # Limit chunks
        chunks_to_use = retrieval_results[:self.max_chunks_in_prompt]
        
        # Build system instruction
        instruction = custom_instruction or self._get_system_instruction()
        
        # Build context section
        context = self._build_context_section(chunks_to_use)
        
        # Build complete prompt
        prompt = f"""{instruction}

# CODE CONTEXT

{context}

# QUESTION

{question}

# ANSWER

"""
        return prompt
    
    def _get_system_instruction(self) -> str:
        """
        Get system instruction based on style.
        
        Returns:
            str: System instruction
        """
        if self.instruction_style == "strict":
            return """You are a code assistant that answers questions based ONLY on the provided code context.

CRITICAL RULES:
1. Answer ONLY using information from the code context below
2. Always cite the specific file path, function/class name, and line numbers
3. If the answer is not in the provided context, explicitly say "I cannot find this information in the provided code"
4. Do NOT make assumptions or use external knowledge
5. Do NOT hallucinate function names, file paths, or code that doesn't exist in the context
6. Be precise and reference specific code locations

When citing code:
- Format: "In `path/to/file.py`, the function `function_name` (lines X-Y)..."
- Always include the repository name if available
- Quote relevant code snippets when helpful"""
        
        else:  # flexible
            return """You are a helpful code assistant. Answer the user's question based on the provided code context.

Guidelines:
- Prioritize information from the code context
- Cite specific files, functions, and line numbers when possible
- If the exact answer isn't in the context, say so clearly
- You may provide general programming knowledge if helpful, but distinguish it from the specific codebase information"""
    
    def _build_context_section(
        self,
        retrieval_results: List[RetrievalResult]
    ) -> str:
        """
        Build context section from retrieval results.
        
        Args:
            retrieval_results: Retrieved chunks
            
        Returns:
            str: Formatted context section
        """
        if not retrieval_results:
            return "No relevant code found."
        
        context_parts = []
        
        for i, result in enumerate(retrieval_results, 1):
            chunk_header = self._format_chunk_header(result, i)
            chunk_code = self._format_chunk_code(result)
            
            context_parts.append(f"{chunk_header}\n\n{chunk_code}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _format_chunk_header(
        self,
        result: RetrievalResult,
        index: int
    ) -> str:
        """
        Format chunk header with metadata.
        
        Args:
            result: Retrieval result
            index: Chunk index
            
        Returns:
            str: Formatted header
        """
        header_parts = [f"## Code Chunk {index}"]
        
        # Repository
        if result.repository_name:
            header_parts.append(f"**Repository:** {result.repository_name}")
        
        # File path
        header_parts.append(f"**File:** `{result.file_path}`")
        
        # Chunk type and name
        if self.include_chunk_types:
            chunk_type = result.chunk_type.capitalize()
            header_parts.append(f"**Type:** {chunk_type}")
            header_parts.append(f"**Name:** `{result.chunk_name}`")
        
        # Line numbers
        if self.include_line_numbers:
            header_parts.append(
                f"**Lines:** {result.line_start}-{result.line_end}"
            )
        
        # Language
        header_parts.append(f"**Language:** {result.language}")
        
        # Relevance score
        header_parts.append(f"**Relevance Score:** {result.combined_score:.3f}")
        
        return "\n".join(header_parts)
    
    def _format_chunk_code(self, result: RetrievalResult) -> str:
        """
        Format chunk code with optional line numbers.
        
        Args:
            result: Retrieval result
            
        Returns:
            str: Formatted code
        """
        code = result.text
        language = result.language.lower()
        
        if self.include_line_numbers and result.line_start:
            # Add line numbers
            lines = code.split('\n')
            numbered_lines = []
            for i, line in enumerate(lines):
                line_num = result.line_start + i
                numbered_lines.append(f"{line_num:4d} | {line}")
            code = '\n'.join(numbered_lines)
        
        return f"```{language}\n{code}\n```"
    
    def build_summary_prompt(
        self,
        question: str,
        retrieval_results: List[RetrievalResult]
    ) -> str:
        """
        Build a prompt for summarizing multiple code chunks.
        
        Args:
            question: User's question
            retrieval_results: Retrieved chunks
            
        Returns:
            str: Summary prompt
        """
        chunks_to_use = retrieval_results[:self.max_chunks_in_prompt]
        
        # List all relevant locations
        locations = []
        for result in chunks_to_use:
            loc = f"- `{result.file_path}` ({result.chunk_type}: `{result.chunk_name}`, lines {result.line_start}-{result.line_end})"
            locations.append(loc)
        
        locations_text = "\n".join(locations)
        
        prompt = f"""You are a code assistant. The user asked: "{question}"

The following code locations are relevant:

{locations_text}

Based on these locations, provide a concise summary that answers the user's question. Cite specific files and line numbers in your answer.
"""
        return prompt
    
    def get_stats(self) -> dict:
        """
        Get prompt builder statistics.
        
        Returns:
            dict: Configuration info
        """
        return {
            "include_line_numbers": self.include_line_numbers,
            "include_chunk_types": self.include_chunk_types,
            "max_chunks_in_prompt": self.max_chunks_in_prompt,
            "instruction_style": self.instruction_style,
        }
