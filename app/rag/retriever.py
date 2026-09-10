"""
Hybrid retriever for code chunks.

This retriever combines vector search, keyword scoring, structural reranking,
and related chunk expansion so the LLM gets the most relevant executable code.
"""
from typing import Dict, List, Optional
import math
import re
import time
from collections import Counter, defaultdict

from app.core import logger, settings
from app.rag.vector_store import VectorStore, ChromaVectorStore
from app.rag.embedding_models import EmbeddingModel, create_embedding_model
from app.rag.retrieval_models import RetrievalResult, RetrievalFilters, RetrievalContext


class CodeRetriever:
    """
    Hybrid retriever for code chunks.

    Features:
    - Hybrid search (vector + keyword/BM25-style scoring)
    - Metadata-aware filtering (language, repo, file, chunk type)
    - Structure-aware reranking for code identifiers and constructors
    - Source-file preference over tests
    - Related class/method expansion for richer class context
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        hybrid_alpha: Optional[float] = None,
        min_similarity: Optional[float] = None,
        enable_reranking: Optional[bool] = None,
    ):
        self.vector_store = vector_store or self._create_default_vector_store()
        self.embedding_model = embedding_model or create_embedding_model()
        self.top_k = top_k or settings.retrieval_top_k
        self.rerank_top_k = rerank_top_k or settings.rerank_top_k
        self.hybrid_alpha = hybrid_alpha if hybrid_alpha is not None else settings.hybrid_alpha
        self.min_similarity = min_similarity if min_similarity is not None else settings.min_similarity_score
        if enable_reranking is None:
            enable_reranking = settings.enable_reranking
        self.enable_reranking = enable_reranking

        logger.info(
            f"CodeRetriever initialized: "
            f"top_k={self.top_k}, "
            f"rerank_top_k={self.rerank_top_k}, "
            f"hybrid_alpha={self.hybrid_alpha}, "
            f"reranking={'enabled' if self.enable_reranking else 'disabled'}"
        )

    def retrieve(
        self,
        query: str,
        filters: Optional[RetrievalFilters] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalContext:
        """
        Retrieve relevant code chunks for a query.

        Pipeline:
        1. Embed query
        2. Retrieve vector candidates
        3. Retrieve keyword candidates
        4. Merge and rerank with code-aware heuristics
        5. Expand related class context
        6. Return final top-k results
        """
        start_time = time.time()
        top_k = top_k or self.top_k
        context = RetrievalContext(query=query, filters=filters)

        logger.info(f"Retrieving for query: {query[:100]}...")

        try:
            query_embedding = self._embed_query(query)

            vector_candidates = self._vector_search(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k * 4,
            )
            keyword_candidates = self._keyword_search(
                query=query,
                filters=filters,
                top_k=top_k * 4,
            )

            candidates = self._merge_candidates(vector_candidates, keyword_candidates)
            context.total_candidates = len(candidates)

            if not candidates:
                logger.warning("No candidates found for query")
                context.retrieval_time_seconds = time.time() - start_time
                return context

            if self.enable_reranking:
                candidates = self._rerank_results(query, candidates)
                context.reranked_count = len(candidates)

            ranked_candidates = candidates[:]
            candidates = [
                candidate for candidate in candidates
                if (candidate.combined_score or 0.0) >= self.min_similarity
            ]
            if not candidates:
                candidates = ranked_candidates[:self.rerank_top_k]

            candidates = self._expand_related_chunks(candidates, filters, query)
            if self.enable_reranking:
                candidates = self._rerank_results(query, candidates)

            context.results = candidates[:self.rerank_top_k]
            context.retrieval_time_seconds = time.time() - start_time

            logger.info(
                f"Retrieved {len(context.results)} results "
                f"from {context.total_candidates} candidates "
                f"in {context.retrieval_time_seconds:.2f}s"
            )
            return context

        except Exception as exc:
            logger.error(f"Retrieval failed: {exc}")
            context.retrieval_time_seconds = time.time() - start_time
            return context

    def retrieve_similar_chunks(
        self,
        chunk_id: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Reserved for future similar-chunk retrieval."""
        logger.warning("retrieve_similar_chunks not yet implemented")
        return []

    def _embed_query(self, query: str) -> List[float]:
        embeddings = self.embedding_model.encode(
            [query],
            batch_size=1,
            normalize=True,
        )
        return embeddings[0].tolist()

    def _vector_search(
        self,
        query_embedding: List[float],
        filters: Optional[RetrievalFilters],
        top_k: int,
    ) -> List[RetrievalResult]:
        try:
            where_filter = None
            if filters and filters.has_filters():
                where_filter = filters.to_chroma_filter()
                logger.info(f"Applying metadata filters: {where_filter}")

            if not hasattr(self.vector_store, "collection"):
                logger.error("Vector store doesn't support querying")
                return []

            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            retrieval_results = []
            if results["ids"] and results["ids"][0]:
                for index in range(len(results["ids"][0])):
                    chunk_id = results["ids"][0][index]
                    document = results["documents"][0][index]
                    distance = results["distances"][0][index]
                    metadata = results["metadatas"][0][index] if results["metadatas"] else {}
                    vector_score = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)
                    retrieval_results.append(
                        self._create_result(
                            chunk_id=chunk_id,
                            text=document,
                            metadata=metadata or {},
                            vector_score=vector_score,
                            bm25_score=0.0,
                        )
                    )

            return retrieval_results

        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")
            return []

    def _keyword_search(
        self,
        query: str,
        filters: Optional[RetrievalFilters],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Retrieve keyword candidates with BM25-style scoring.

        This helps code-specific identifiers like class names and file paths that
        vector search alone often treats too loosely.
        """
        if not hasattr(self.vector_store, "collection"):
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        try:
            where_filter = filters.to_chroma_filter() if filters and filters.has_filters() else None
            raw = self.vector_store.collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning(f"Keyword search failed: {exc}")
            return []

        ids = raw.get("ids", [])
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])
        if not ids:
            return []

        tokenized_docs = []
        doc_freq: Dict[str, int] = defaultdict(int)
        for document, metadata in zip(documents, metadatas):
            keyword_text = self._build_keyword_text(document, metadata or {})
            tokens = self._tokenize(keyword_text)
            tokenized_docs.append(tokens)
            for token in set(tokens):
                doc_freq[token] += 1

        avg_doc_len = (
            sum(len(tokens) for tokens in tokenized_docs) / len(tokenized_docs)
            if tokenized_docs else 1.0
        )

        scored_documents = []
        for index, tokens in enumerate(tokenized_docs):
            bm25_score = self._bm25_score(
                query_terms=query_terms,
                doc_terms=tokens,
                doc_freq=doc_freq,
                total_docs=len(tokenized_docs),
                avg_doc_len=avg_doc_len,
            )
            exact_score = self._exact_identifier_score(query, metadatas[index] or {})
            intent_score = self._intent_score(query, metadatas[index] or {})
            score = bm25_score + exact_score + intent_score
            if score > 0:
                scored_documents.append((index, score))

        if not scored_documents:
            return []

        max_score = max(score for _, score in scored_documents) or 1.0
        results = []
        for index, score in sorted(scored_documents, key=lambda item: item[1], reverse=True)[:top_k]:
            metadata = metadatas[index] or {}
            normalized_score = min(score / max_score, 1.0)
            results.append(
                self._create_result(
                    chunk_id=ids[index],
                    text=documents[index],
                    metadata=metadata,
                    vector_score=0.0,
                    bm25_score=normalized_score,
                )
            )

        return results

    def _merge_candidates(
        self,
        vector_candidates: List[RetrievalResult],
        keyword_candidates: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        merged: Dict[str, RetrievalResult] = {}

        for result in vector_candidates + keyword_candidates:
            existing = merged.get(result.chunk_id)
            if existing is None:
                merged[result.chunk_id] = result
                continue

            existing.vector_score = max(existing.vector_score or 0.0, result.vector_score or 0.0)
            existing.bm25_score = max(existing.bm25_score or 0.0, result.bm25_score or 0.0)
            existing.combined_score = (
                self.hybrid_alpha * (existing.vector_score or 0.0) +
                (1.0 - self.hybrid_alpha) * (existing.bm25_score or 0.0)
            )

        return sorted(
            merged.values(),
            key=lambda result: result.combined_score or 0.0,
            reverse=True,
        )

    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        query_terms = set(self._tokenize(query))
        constructor_query = self._is_constructor_query(query)

        for result in results:
            base_score = result.combined_score or result.vector_score or 0.0
            metadata = result.metadata or {}
            chunk_name = str(result.chunk_name or "")
            name_terms = set(self._tokenize(chunk_name))

            if name_terms:
                overlap = len(query_terms & name_terms)
                if overlap:
                    base_score *= 1.0 + (0.18 * overlap)

            exact_identifier_score = self._exact_identifier_score(query, metadata)
            if exact_identifier_score:
                base_score *= 1.0 + min(exact_identifier_score * 0.25, 1.0)

            base_score *= self._chunk_type_boost(result.chunk_type, constructor_query, chunk_name)
            base_score *= self._source_file_boost(result.file_path or "")
            base_score *= 1.0 + self._intent_score(query, metadata)

            if metadata.get("docstring"):
                base_score *= 1.05

            result.rerank_score = min(base_score, 10.0)
            result.combined_score = min(base_score, 10.0)

        results.sort(
            key=lambda result: (
                result.rerank_score or 0.0,
                result.bm25_score or 0.0,
                result.vector_score or 0.0,
            ),
            reverse=True,
        )
        return results

    def _expand_related_chunks(
        self,
        results: List[RetrievalResult],
        filters: Optional[RetrievalFilters],
        query: str,
    ) -> List[RetrievalResult]:
        """
        Add nearby semantic chunks for retrieved classes and methods.

        This is especially useful when the user asks about a class constructor or
        a method's responsibility after the class itself was already identified.
        """
        if not hasattr(self.vector_store, "collection"):
            return results

        merged: Dict[str, RetrievalResult] = {result.chunk_id: result for result in results}
        constructor_query = self._is_constructor_query(query)
        anchors = [
            result for result in results[: self.rerank_top_k]
            if result.chunk_type in {"class", "class_header", "method"}
        ]

        for anchor in anchors:
            if not anchor.file_path:
                continue

            class_name = self._class_name_for(anchor)
            try:
                raw = self.vector_store.collection.get(
                    where=self._related_where(anchor, filters),
                    include=["documents", "metadatas"],
                )
            except Exception as exc:
                logger.debug(f"Related chunk lookup failed for {anchor.file_path}: {exc}")
                continue

            for chunk_id, document, metadata in zip(
                raw.get("ids", []),
                raw.get("documents", []),
                raw.get("metadatas", []),
            ):
                metadata = metadata or {}
                if chunk_id in merged:
                    continue
                if not self._is_related_to_anchor(metadata, anchor, class_name, constructor_query):
                    continue

                related = self._create_result(
                    chunk_id=chunk_id,
                    text=document,
                    metadata=metadata,
                    vector_score=(anchor.vector_score or 0.0) * 0.85,
                    bm25_score=(anchor.bm25_score or 0.0) * 0.85,
                )
                related.combined_score = (anchor.combined_score or 0.0) * 0.85
                merged[chunk_id] = related

                if len(merged) >= self.rerank_top_k * 3:
                    break

        return list(merged.values())

    def _related_where(
        self,
        anchor: RetrievalResult,
        filters: Optional[RetrievalFilters],
    ) -> Dict[str, object]:
        conditions = [{"file_path": anchor.file_path}]

        if filters and filters.repository_id:
            conditions.append({"repository_id": filters.repository_id})
        elif filters and filters.repository_name:
            conditions.append({"repository_name": filters.repository_name})
        elif anchor.repository_id:
            conditions.append({"repository_id": anchor.repository_id})
        elif anchor.repository_name:
            conditions.append({"repository_name": anchor.repository_name})

        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def _is_related_to_anchor(
        self,
        metadata: Dict[str, object],
        anchor: RetrievalResult,
        class_name: Optional[str],
        constructor_query: bool,
    ) -> bool:
        if metadata.get("file_path") != anchor.file_path:
            return False

        chunk_type = metadata.get("chunk_type")
        chunk_name = str(metadata.get("chunk_name") or "")
        metadata_class_name = str(metadata.get("class_name") or "")
        method_names = str(metadata.get("method_names") or "")

        if chunk_type in {"import", "module", "text_block"}:
            return False

        if class_name:
            if constructor_query and chunk_name.endswith(".__init__"):
                return True
            return (
                chunk_name == class_name or
                chunk_name.startswith(f"{class_name}.") or
                metadata_class_name == class_name or
                class_name in method_names
            )

        return chunk_type in {"function", "method", "class", "class_header"}

    def _class_name_for(self, result: RetrievalResult) -> Optional[str]:
        if result.chunk_type in {"class", "class_header"} and result.chunk_name:
            return str(result.chunk_name).replace("_header", "")
        if result.chunk_type == "method" and result.chunk_name and "." in str(result.chunk_name):
            return str(result.chunk_name).split(".", 1)[0]
        class_name = result.metadata.get("class_name")
        return str(class_name) if class_name else None

    def _create_result(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, object],
        vector_score: float,
        bm25_score: float,
    ) -> RetrievalResult:
        combined_score = self.hybrid_alpha * vector_score + (1.0 - self.hybrid_alpha) * bm25_score
        return RetrievalResult(
            chunk_id=chunk_id,
            text=text,
            vector_score=vector_score,
            bm25_score=bm25_score,
            combined_score=combined_score,
            repository_name=metadata.get("repository_name"),
            repository_id=metadata.get("repository_id"),
            file_path=metadata.get("file_path"),
            language=metadata.get("language"),
            chunk_type=metadata.get("chunk_type"),
            chunk_name=metadata.get("chunk_name"),
            line_start=metadata.get("line_start"),
            line_end=metadata.get("line_end"),
            metadata=metadata,
        )

    def _build_keyword_text(self, document: str, metadata: Dict[str, object]) -> str:
        parts = [
            str(metadata.get("repository_name") or ""),
            str(metadata.get("file_path") or ""),
            str(metadata.get("chunk_type") or ""),
            str(metadata.get("chunk_name") or ""),
            str(metadata.get("class_name") or ""),
            str(metadata.get("method_names") or ""),
            str(metadata.get("docstring") or ""),
            document,
        ]
        return "\n".join(parts)

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(
            r"[A-Za-z0-9_./\\-]+\.\w+|[A-Za-z_][A-Za-z0-9_]*",
            text.lower(),
        )
        expanded = []
        for token in tokens:
            expanded.append(token)
            expanded.extend(
                part for part in re.split(r"[._/\\-]+", token)
                if len(part) > 2
            )
        return expanded

    def _bm25_score(
        self,
        query_terms: List[str],
        doc_terms: List[str],
        doc_freq: Dict[str, int],
        total_docs: int,
        avg_doc_len: float,
    ) -> float:
        if not doc_terms:
            return 0.0

        term_counts = Counter(doc_terms)
        doc_len = len(doc_terms)
        k1 = 1.5
        b = 0.75
        score = 0.0

        for term in set(query_terms):
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1.0))
            score += idf * (tf * (k1 + 1)) / denominator

        return score

    def _exact_identifier_score(self, query: str, metadata: Dict[str, object]) -> float:
        query_lower = query.lower()
        identifiers = [
            metadata.get("chunk_name"),
            metadata.get("class_name"),
            metadata.get("file_path"),
        ]
        identifiers.extend(str(metadata.get("method_names") or "").split(","))

        score = 0.0
        for identifier in identifiers:
            identifier_text = str(identifier or "").strip()
            if not identifier_text:
                continue
            identifier_lower = identifier_text.lower()
            short_name = identifier_lower.split(".")[-1]
            if identifier_lower in query_lower:
                score += 1.0
            elif short_name and short_name in query_lower:
                score += 0.7

        return score

    def _intent_score(self, query: str, metadata: Dict[str, object]) -> float:
        """
        Boost special code intents like constructor queries.

        This helps follow-ups such as "What is its constructor responsible for?"
        land on `Class.__init__` rather than arbitrary sibling methods.
        """
        chunk_name = str(metadata.get("chunk_name") or "")
        chunk_type = str(metadata.get("chunk_type") or "")
        score = 0.0

        if self._is_constructor_query(query):
            if chunk_name.endswith(".__init__"):
                score += 1.4
            if chunk_type in {"class", "class_header"}:
                score += 0.35
            if "__init__" in chunk_name:
                score += 0.5

        if self._is_method_responsibility_query(query):
            if chunk_type in {"function", "method"}:
                score += 0.2

        return score

    def _chunk_type_boost(
        self,
        chunk_type: Optional[str],
        constructor_query: bool,
        chunk_name: str,
    ) -> float:
        boost = {
            "function": 1.2,
            "method": 1.25,
            "class": 1.12,
            "class_header": 1.08,
            "module": 0.92,
            "import": 0.72,
            "text_block": 0.68,
        }.get(chunk_type or "", 1.0)

        if constructor_query and chunk_name.endswith(".__init__"):
            boost *= 1.6

        return boost

    def _source_file_boost(self, file_path: str) -> float:
        if not file_path:
            return 1.0
        if self._is_test_file(file_path):
            return 0.78
        return 1.08

    def _is_constructor_query(self, query: str) -> bool:
        query_lower = query.lower()
        constructor_terms = [
            "constructor",
            "__init__",
            "init method",
            "initializ",
            "instantiat",
            "created",
            "creation",
            "set up",
            "setup",
        ]
        return any(term in query_lower for term in constructor_terms)

    def _is_method_responsibility_query(self, query: str) -> bool:
        query_lower = query.lower()
        phrases = [
            "responsible for",
            "what does",
            "how does",
            "purpose of",
            "role of",
        ]
        return any(phrase in query_lower for phrase in phrases)

    def _is_test_file(self, file_path: str) -> bool:
        normalized = file_path.replace("\\", "/").lower()
        return (
            "/test/" in normalized or
            "/tests/" in normalized or
            normalized.startswith("test/") or
            normalized.startswith("tests/") or
            normalized.endswith("_test.py") or
            normalized.endswith("_test.js") or
            normalized.endswith(".test.js") or
            normalized.endswith(".spec.js") or
            normalized.startswith("test_")
        )

    def _create_default_vector_store(self) -> VectorStore:
        return ChromaVectorStore(
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
            distance_metric=settings.chroma_distance_metric,
        )

    def get_stats(self) -> dict:
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
