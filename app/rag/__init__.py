"""
RAG pipeline components for code understanding and retrieval.

This package exposes the main pipeline types while tolerating optional
dependencies during lightweight retrieval-only checks.
"""
from importlib import import_module


__all__ = []


def _export(module_name: str, names: list[str]) -> None:
    """Best-effort export so optional deps do not break unrelated imports."""
    try:
        module = import_module(module_name)
    except Exception:
        return

    for name in names:
        if hasattr(module, name):
            globals()[name] = getattr(module, name)
            __all__.append(name)


_export("app.rag.parser", ["CodeParser"])
_export("app.rag.chunk_models", ["CodeChunk", "ChunkingResult"])
_export("app.rag.base_chunker", ["BaseChunker", "TextChunker"])
_export("app.rag.ast_chunker", ["ASTChunker"])
_export(
    "app.rag.embedding_models",
    ["EmbeddingModel", "SentenceTransformerModel", "HuggingFaceModel", "create_embedding_model"],
)
_export("app.rag.embedder", ["CodeEmbedder", "EmbeddingCache"])
_export("app.rag.vector_store", ["VectorStore", "ChromaVectorStore"])
_export("app.rag.indexer", ["CodeIndexer", "IndexingStats"])
_export("app.rag.retrieval_models", ["RetrievalResult", "RetrievalFilters", "RetrievalContext"])
_export("app.rag.retriever", ["CodeRetriever"])
_export(
    "app.rag.llm_client",
    [
        "LLMClient",
        "LLMResponse",
        "HuggingFaceLLMClient",
        "GroqLLMClient",
        "OpenAILLMClient",
        "OllamaLLMClient",
        "create_llm_client",
    ],
)
_export("app.rag.prompt_builder", ["PromptBuilder"])
_export("app.rag.rag_chain", ["RAGChain", "RAGResponse"])
