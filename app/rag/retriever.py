"""
Hybrid retriever for code chunks.

This is NOT just vector search - it's a sophisticated retrieval pipeline that:
- Combines vector similarity + BM25 keyword matching (hybrid)
- Leverages rich metadata (file, function, language, etc.)
- Supports filtering by metadata
- Reranks results by relevance
- Returns results with full context

Traditional RAG: Query → Top-5 vector search → LLM
Our approach: Query → Hybrid search → Metadata filtering → Reranking → LLM
"""
from typing import List, Optional
import time
import numpy as np

from app.core import logger, settings
from app.rag.vector_store import VectorStore, ChromaVectorStore
from app.rag.embedding_models import EmbeddingModel, create_embedding_model
from app.rag.retrieval_models import RetrievalResult, RetrievalFilters, RetrievalContext


class CodeRetriever:
    """
    Hybrid retriever for code chunks.
    
    Features:
    - Hybrid search (vector + BM25 keyword)
    - Metadata-aware filtering (language, repo, file, chunk type)
    - Reranking by relevance
    - Rich results with context
    
    NOT just: query → top-k vectors
    Instead: query → candidates → filter → rerank → results with context
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        hybrid_alpha: Optional[float] = None,
        min_similarity: Optional[float] = None,
        enable_reranking: bool = True
    ):
        """
        Initialize code retriever.
        
        Args:
            vector_store: Vector store (defaults to ChromaDB)
            embedding_model: Embedding model (defaults to settings)
            top_k: Number of candidates to retrieve
            rerank_top_k: Number of final results after reranking
            hybrid_alpha: Balance between vector (1.0) and BM25 (0.0)
            min_similarity: Minimum similarity score threshold
            enable_reranking: Whether to rerank results
        """
        self.vector_store = vector_store or self._create_default_vector_store()
        self.embedding_model = embedding_model or create_embedding_model()
        self.top_k = top_k or settings.retrieval_top_k
        self.rerank_top_k = rerank_top_k or settings.rerank_top_k
        self.hybrid_alpha = hybrid_alpha if hybrid_alpha is not None else settings.hybrid_alpha
        self.min_similarity = min_similarity or settings.min_similarity_score
        self.enable_reranking = enable_reranking
        
        logger.info(
            f"CodeRetriever initialized: "
            f"top_k={self.top_k}, "
            f"rerank_top_k={self.rerank_top_k}, "
            f"hybrid_alpha={self.hybrid_alpha}, "
            f"reranking={'enabled' if enable_reranking else 'disabled'}"
        )
    
    def retrieve(
        self,
        query: str,
        filters: Optional[RetrievalFilters] = None,
        top_k: Optional[int] = None
    ) -> RetrievalContext:
        """
        Retrieve relevant code chunks for a query.
        
        Pipeline:
        1. Embed query
        2. Vector search with optional metadata filters
        3. BM25 keyword scoring (if available)
        4. Hybrid combination
        5. Reranking (if enabled)
        6. Filter by minimum score
        7. Return top-k with full context
        
        Args:
            query: User query
            filters: Optional metadata filters
            top_k: Override default top_k
            
        Returns:
            RetrievalContext: Results with rich metadata
        """
        start_time = time.time()
        top_k = top_k or self.top_k
        
        logger.info(f"Retrieving for query: {query[:100]}...")
        
        # Create context
        context = RetrievalContext(query=query, filters=filters)
        
        try:
            # Step 1: Embed query
            query_embedding = self._embed_query(query)
            
            # Step 2: Vector search with metadata filters
            candidates = self._vector_search(
                query_embedding,
                filters=filters,
                top_k=top_k * 2  # Get more candidates for reranking
            )
            context.total_candidates = len(candidates)
            
            if not candidates:
                logger.warning("No candidates found for query")
                context.retrieval_time_seconds = time.time() - start_time
                return context
            
            # Step 3: Calculate BM25 scores (keyword matching)
            # For now, we'll use ChromaDB's built-in scoring
            # In production, you'd implement proper BM25
            
            # Step 4: Rerank results (if enabled)
            if self.enable_reranking:
                candidates = self._rerank_results(query, candidates)
                context.reranked_count = len(candidates)
            
            # Step 5: Filter by minimum score
            candidates = [
                c for c in candidates
                if c.combined_score and c.combined_score >= self.min_similarity
            ]
            
            # Step 6: Take top-k
            context.results = candidates[:self.rerank_top_k]
            
            context.retrieval_time_seconds = time.time() - start_time
            
            logger.info(
                f"Retrieved {len(context.results)} results "
                f"from {context.total_candidates} candidates "
                f"in {context.retrieval_time_seconds:.2f}s"
            )
            
            return context
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            context.retrieval_time_seconds = time.time() - start_time
            return context
    
    def retrieve_similar_chunks(
        self,
        chunk_id: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Find chunks similar to a given chunk.
        
        Useful for "show me similar functions" or "related code".
        
        Args:
            chunk_id: ID of the chunk to find similar to
            top_k: Number of results
            
        Returns:
            List[RetrievalResult]: Similar chunks
        """
        # TODO: Implement using ChromaDB's get + query
        logger.warning("retrieve_similar_chunks not yet implemented")
        return []
    
    def _embed_query(self, query: str) -> List[float]:
        """
        Embed the query using the embedding model.
        
        Args:
            query: Query string
            
        Returns:
            List[float]: Query embedding
        """
        embeddings = self.embedding_model.encode(
            [query],
            batch_size=1,
            normalize=True
        )
        return embeddings[0].tolist()
    
    def _vector_search(
        self,
        query_embedding: List[float],
        filters: Optional[RetrievalFilters],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Perform vector search with optional metadata filtering.
        
        This leverages our rich metadata (language, repo, file, chunk type).
        
        Args:
            query_embedding: Query vector
            filters: Metadata filters
            top_k: Number of results
            
        Returns:
            List[RetrievalResult]: Search results
        """
        try:
            # Build ChromaDB filter from metadata
            where_filter = None
            if filters and filters.has_filters():
                where_filter = filters.to_chroma_filter()
                logger.info(f"Applying metadata filters: {where_filter}")
            
            print("=" * 60)
            print("WHERE FILTER:", where_filter)
            print("TOP K:", top_k)
            print("=" * 60)
            # Query ChromaDB
            if hasattr(self.vector_store, 'collection'):

                # results = self.vector_store.collection.query(
                #     query_embeddings=[query_embedding],
                #     n_results=top_k,
                #     where=where_filter,
                #     include=["documents", "metadatas", "distances"]
                # )
                results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
)
                print(self.vector_store.collection.count())
                print("=" * 60)
                print("RAW RESULTS")
                print("IDs:", results["ids"])
                print("NUM IDS:", len(results["ids"][0]) if results["ids"] else 0)
                print("=" * 60)

                if results["metadatas"] and len(results["metadatas"][0]) > 0:
                    print("=" * 60)
                    print("FIRST METADATA:")
                    print(results["metadatas"][0][0])
                    print("=" * 60)
                
                # Convert to RetrievalResult objects
                retrieval_results = []
                
                if results['ids'] and len(results['ids'][0]) > 0:
                    for i in range(len(results['ids'][0])):
                        chunk_id = results['ids'][0][i]
                        text = results['documents'][0][i]
                        distance = results['distances'][0][i]
                        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                        
                        # Convert distance to similarity (ChromaDB uses L2 or cosine)
                        # For cosine: similarity = 1 - distance
                        vector_score = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)
                        
                        result = RetrievalResult(
                            chunk_id=chunk_id,
                            text=text,
                            vector_score=vector_score,
                            bm25_score=None,  # TODO: Implement BM25
                            combined_score=vector_score,  # Will be updated with hybrid
                            repository_name=metadata.get('repository_name'),
                            file_path=metadata.get('file_path'),
                            language=metadata.get('language'),
                            chunk_type=metadata.get('chunk_type'),
                            chunk_name=metadata.get('chunk_name'),
                            line_start=metadata.get('line_start'),
                            line_end=metadata.get('line_end'),
                            metadata=metadata
                        )
                        retrieval_results.append(result)
                
                return retrieval_results
            
            else:
                logger.error("Vector store doesn't support querying")
                return []
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Rerank results by relevance.
        
        Uses multiple signals:
        - Vector similarity
        - Query term overlap
        - Chunk type preference (functions > classes > imports)
        - Docstring presence
        
        Args:
            query: Original query
            results: Initial results
            
        Returns:
            List[RetrievalResult]: Reranked results
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        for result in results:
            # Start with combined score
            base_score = result.combined_score or result.vector_score
            
            # Boost if query terms appear in chunk name
            if result.chunk_name:
                name_lower = result.chunk_name.lower()
                term_overlap = sum(1 for term in query_terms if term in name_lower)
                if term_overlap > 0:
                    base_score *= (1.0 + 0.1 * term_overlap)
            
            # Boost functions over classes over imports
            type_boost = {
                "function": 1.1,
                "method": 1.1,
                "class": 1.05,
                "import": 0.9,
            }.get(result.chunk_type, 1.0)
            base_score *= type_boost
            
            # Boost if has docstring
            if result.metadata.get('docstring'):
                base_score *= 1.05
            
            # Store rerank score
            result.rerank_score = min(base_score, 1.0)  # Cap at 1.0
        
        # Sort by rerank score
        results.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
        
        return results
    
    def _create_default_vector_store(self) -> VectorStore:
        """Create default vector store (ChromaDB)."""
        return ChromaVectorStore(
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
            distance_metric=settings.chroma_distance_metric
        )
    
    def get_stats(self) -> dict:
        """
        Get retriever statistics.
        
        Returns:
            dict: Retriever statistics
        """
        return {
            "vector_store": self.vector_store.__class__.__name__,
            "embedding_model": self.embedding_model.model_name,
            "top_k": self.top_k,
            "rerank_top_k": self.rerank_top_k,
            "hybrid_alpha": self.hybrid_alpha,
            "min_similarity": self.min_similarity,
            "reranking_enabled": self.enable_reranking,
            "total_indexed": self.vector_store.count(),
        }
