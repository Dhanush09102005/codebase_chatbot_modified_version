"""
RAG pipeline components for code understanding and retrieval.

Complete pipeline:
  Parser → Chunker → Embedder → Indexer → Retriever → LLM Chain → Answer
"""
from app.rag.parser import CodeParser
from app.rag.chunk_models import CodeChunk, ChunkingResult
from app.rag.base_chunker import BaseChunker, TextChunker
from app.rag.ast_chunker import ASTChunker
from app.rag.embedding_models import (
    EmbeddingModel,
    SentenceTransformerModel,
    HuggingFaceModel,
    create_embedding_model,
)
from app.rag.embedder import CodeEmbedder, EmbeddingCache
from app.rag.vector_store import VectorStore, ChromaVectorStore
from app.rag.indexer import CodeIndexer, IndexingStats
from app.rag.retrieval_models import RetrievalResult, RetrievalFilters, RetrievalContext
from app.rag.retriever import CodeRetriever
from app.rag.llm_client import (
    LLMClient,
    LLMResponse,
    HuggingFaceLLMClient,
    GroqLLMClient,
    OpenAILLMClient,
    OllamaLLMClient,
    create_llm_client,
)
from app.rag.prompt_builder import PromptBuilder
from app.rag.rag_chain import RAGChain, RAGResponse

__all__ = [
    # Parser
    "CodeParser",
    # Chunker
    "CodeChunk",
    "ChunkingResult",
    "BaseChunker",
    "TextChunker",
    "ASTChunker",
    # Embedder
    "EmbeddingModel",
    "SentenceTransformerModel",
    "HuggingFaceModel",
    "create_embedding_model",
    "CodeEmbedder",
    "EmbeddingCache",
    # Vector Store & Indexer
    "VectorStore",
    "ChromaVectorStore",
    "CodeIndexer",
    "IndexingStats",
    # Retriever
    "RetrievalResult",
    "RetrievalFilters",
    "RetrievalContext",
    "CodeRetriever",
    # LLM Client
    "LLMClient",
    "LLMResponse",
    "HuggingFaceLLMClient",
    "GroqLLMClient",
    "OpenAILLMClient",
    "OllamaLLMClient",
    "create_llm_client",
    # Prompt Builder
    "PromptBuilder",
    # RAG Chain
    "RAGChain",
    "RAGResponse",
]
