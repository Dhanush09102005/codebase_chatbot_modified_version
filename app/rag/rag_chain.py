"""
RAG Chain - Orchestrator for the complete RAG workflow.

The RAG chain ONLY orchestrates the workflow.
It does NOT contain retrieval logic, embedding logic, or vector database operations.

Workflow:
1. Accept user's question
2. Call retriever to get relevant chunks
3. Call prompt builder to format prompt
4. Call LLM client to generate answer
5. Return answer with metadata
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import time

from app.core import logger
from app.rag.retriever import CodeRetriever
from app.rag.retrieval_models import RetrievalFilters, RetrievalContext
from app.rag.prompt_builder import PromptBuilder
from app.rag.llm_client import LLMClient, LLMResponse


@dataclass
class RAGResponse:
    """
    Complete RAG response with all metadata.
    
    This includes the answer, retrieval context, and timing information.
    """
    question: str  # Original question
    answer: str  # Generated answer
    retrieval_context: RetrievalContext  # Retrieval results and stats
    llm_response: LLMResponse  # Full LLM response with tokens
    total_time_seconds: float  # Total time for complete workflow
    prompt_used: str  # The actual prompt sent to LLM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "retrieval_stats": {
                "chunks_retrieved": len(self.retrieval_context.results),
                "repositories": list(set(
                    r.repository_name for r in self.retrieval_context.results
                    if r.repository_name
                )),
                "files": list(set(
                    r.file_path for r in self.retrieval_context.results
                )),
                "retrieval_time_seconds": self.retrieval_context.retrieval_time_seconds,
            },
            "llm_stats": {
                "model": self.llm_response.model,
                "tokens_used": self.llm_response.tokens_used,
                "prompt_tokens": self.llm_response.prompt_tokens,
                "completion_tokens": self.llm_response.completion_tokens,
                "llm_time_seconds": self.llm_response.time_seconds,
            },
            "total_time_seconds": self.total_time_seconds,
            "metadata": self.metadata,
        }
    
    def format_for_display(self) -> str:
        """
        Format response for human-readable display.
        
        Returns:
            str: Formatted response
        """
        output = []
        
        output.append("=" * 80)
        output.append("QUESTION")
        output.append("=" * 80)
        output.append(self.question)
        output.append("")
        
        output.append("=" * 80)
        output.append("ANSWER")
        output.append("=" * 80)
        output.append(self.answer)
        output.append("")
        
        output.append("=" * 80)
        output.append("SOURCES")
        output.append("=" * 80)
        for i, result in enumerate(self.retrieval_context.results, 1):
            output.append(
                f"{i}. {result.file_path} "
                f"({result.chunk_type}: {result.chunk_name}, "
                f"lines {result.line_start}-{result.line_end}, "
                f"score: {(result.combined_score or 0):.3f})"
            )
        output.append("")
        
        output.append("=" * 80)
        output.append("STATISTICS")
        output.append("=" * 80)
        output.append(f"Total Time: {self.total_time_seconds:.2f}s")
        output.append(f"  Retrieval: {self.retrieval_context.retrieval_time_seconds:.2f}s")
        output.append(f"  LLM: {self.llm_response.time_seconds:.2f}s")
        output.append(f"Chunks Retrieved: {len(self.retrieval_context.results)}")
        output.append(f"Tokens Used: {self.llm_response.tokens_used}")
        output.append(f"Model: {self.llm_response.model}")
        
        return "\n".join(output)


class RAGChain:
    """
    RAG Chain - Orchestrates the complete RAG workflow.
    
    This is the ORCHESTRATOR. It connects components but doesn't
    contain their logic:
    - Retriever: finds relevant chunks
    - PromptBuilder: formats prompts
    - LLMClient: generates answers
    
    The chain is stateless - each query is independent.
    """
    
    def __init__(
        self,
        retriever: Optional[CodeRetriever] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialize RAG chain.
        
        Args:
            retriever: Code retriever (creates default if None)
            prompt_builder: Prompt builder (creates default if None)
            llm_client: LLM client (creates default if None)
        """
        from app.rag.llm_client import create_llm_client
        
        self.retriever = retriever or CodeRetriever()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_client = llm_client or create_llm_client(provider="huggingface")
        
        logger.info(
            f"RAGChain initialized: "
            f"retriever={type(self.retriever).__name__}, "
            f"llm={self.llm_client.model_name}"
        )
    
    def query(
        self,
        question: str,
        filters: Optional[RetrievalFilters] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        custom_instruction: Optional[str] = None,
    ) -> RAGResponse:
        """
        Execute complete RAG workflow.
        
        Workflow:
        1. Retrieve relevant code chunks (Retriever)
        2. Build structured prompt (PromptBuilder)
        3. Generate answer (LLMClient)
        4. Return complete response
        
        Args:
            question: User's question
            filters: Optional retrieval filters
            max_tokens: Maximum tokens for LLM generation
            temperature: LLM sampling temperature
            custom_instruction: Custom system instruction
            
        Returns:
            RAGResponse: Complete response with metadata
        """
        logger.info(f"RAG query: {question[:100]}...")
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant chunks
            logger.info("Step 1/3: Retrieving relevant code chunks...")
            retrieval_context = self.retriever.retrieve(
                query=question,
                filters=filters
            )
            
            logger.info(
                f"Retrieved {len(retrieval_context.results)} chunks "
                f"in {retrieval_context.retrieval_time_seconds:.2f}s"
            )
            
            # Check if any chunks found
            if not retrieval_context.results:
                logger.warning("No relevant code chunks found")
                
                # Return empty response
                empty_response = LLMResponse(
                    text="I couldn't find any relevant code to answer your question. Please try rephrasing or check if the repository has been indexed.",
                    model=self.llm_client.model_name,
                    time_seconds=0.0
                )
                
                total_time = time.time() - start_time
                
                return RAGResponse(
                    question=question,
                    answer=empty_response.text,
                    retrieval_context=retrieval_context,
                    llm_response=empty_response,
                    total_time_seconds=total_time,
                    prompt_used="",
                    metadata={"status": "no_chunks_found"}
                )
            
            # Step 2: Build prompt
            logger.info("Step 2/3: Building structured prompt...")
            prompt = self.prompt_builder.build_prompt(
                question=question,
                retrieval_results=retrieval_context.results,
                custom_instruction=custom_instruction
            )
            
            logger.info(f"Prompt built: {len(prompt)} characters")
            
            # Step 3: Generate answer
            logger.info("Step 3/3: Generating answer with LLM...")
            llm_response = self.llm_client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            logger.info(
                f"Answer generated: {len(llm_response.text)} characters, "
                f"{llm_response.tokens_used} tokens"
            )
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Build complete response
            response = RAGResponse(
                question=question,
                answer=llm_response.text,
                retrieval_context=retrieval_context,
                llm_response=llm_response,
                total_time_seconds=total_time,
                prompt_used=prompt,
                metadata={
                    "status": "success",
                    "chunks_used": len(retrieval_context.results),
                }
            )
            
            logger.info(f"RAG query complete: {total_time:.2f}s total")
            
            return response
        
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise
    
    def query_batch(
        self,
        questions: list[str],
        filters: Optional[RetrievalFilters] = None,
        **kwargs
    ) -> list[RAGResponse]:
        """
        Execute multiple RAG queries.
        
        Args:
            questions: List of questions
            filters: Optional retrieval filters
            **kwargs: Additional parameters for query()
            
        Returns:
            list[RAGResponse]: List of responses
        """
        logger.info(f"Batch query: {len(questions)} questions")
        
        responses = []
        for question in questions:
            response = self.query(question, filters=filters, **kwargs)
            responses.append(response)
        
        logger.info(f"Batch query complete: {len(responses)} responses")
        
        return responses
    
    def explain_answer(self, response: RAGResponse) -> str:
        """
        Generate explanation of how the answer was derived.
        
        Args:
            response: RAG response to explain
            
        Returns:
            str: Explanation text
        """
        explanation_parts = []
        
        explanation_parts.append(
            f"To answer the question '{response.question}', "
            f"I performed the following steps:"
        )
        
        explanation_parts.append(
            f"\n1. Retrieved {len(response.retrieval_context.results)} "
            f"relevant code chunks using hybrid search "
            f"(vector similarity + keyword matching + reranking)."
        )
        
        explanation_parts.append("\n2. The most relevant chunks were:")
        for i, result in enumerate(response.retrieval_context.results[:3], 1):
            explanation_parts.append(
                f"   {i}. {result.file_path} - {result.chunk_name} "
                f"(score: {(result.combined_score or 0):.3f})"
            )
        
        explanation_parts.append(
            f"\n3. Built a structured prompt with code context, "
            f"file paths, line numbers, and strict instructions."
        )
        
        explanation_parts.append(
            f"\n4. Generated answer using {response.llm_response.model} "
            f"({response.llm_response.tokens_used} tokens)."
        )
        
        explanation_parts.append(
            f"\n5. Total time: {response.total_time_seconds:.2f}s "
            f"(retrieval: {response.retrieval_context.retrieval_time_seconds:.2f}s, "
            f"LLM: {response.llm_response.time_seconds:.2f}s)"
        )
        
        return "".join(explanation_parts)
    
    def get_stats(self) -> dict:
        """
        Get RAG chain statistics.
        
        Returns:
            dict: Configuration and component info
        """
        return {
            "retriever": type(self.retriever).__name__,
            "prompt_builder": self.prompt_builder.get_stats(),
            "llm_model": self.llm_client.model_name,
        }
