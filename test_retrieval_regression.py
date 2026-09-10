#!/usr/bin/env python3
"""
Targeted retrieval regression checks for the codebase chat pipeline.

This focuses on retrieval quality rather than full LLM generation:
- follow-up query rewriting
- source code vs test file ranking
- identifier-aware ranking
- related class/method expansion
- retrieval latency and diagnostics
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


@dataclass
class RetrievalScenario:
    """A single retrieval evaluation scenario."""

    name: str
    question: str
    expected_chunk_names: List[str] = field(default_factory=list)
    expected_file_paths: List[str] = field(default_factory=list)
    forbidden_file_paths: List[str] = field(default_factory=list)
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    max_latency_seconds: Optional[float] = None


class FakeRetriever:
    """
    Minimal retriever stub for validating RAGChain query rewriting.

    It records the final retrieval query and returns no chunks, which is enough
    to verify whether follow-up questions are rewritten with useful context.
    """

    def __init__(self):
        self.last_query: Optional[str] = None

    def retrieve(self, query: str, filters=None):
        RetrievalContext = _load_retrieval_context()
        self.last_query = query
        return RetrievalContext(query=query, filters=filters)


def evaluate_live_retrieval():
    """Run retrieval checks against the current indexed ChromaDB collection."""
    try:
        CodeRetriever = _load_code_retriever()
        PromptBuilder = _load_prompt_builder()
        RetrievalFilters = _load_retrieval_filters()
        retriever = CodeRetriever()
        prompt_builder = PromptBuilder()
    except Exception as exc:
        print("=" * 80)
        print("LIVE RETRIEVAL UNAVAILABLE")
        print("=" * 80)
        print(f"Could not initialize retriever: {exc}")
        print("The regression harness is ready, but live retrieval needs local dependencies.")
        print()
        return

    scenarios = [
        RetrievalScenario(
            name="Identifier ranking",
            question="How does HTTPAdapter.send work?",
            expected_chunk_names=["HTTPAdapter.send"],
            expected_file_paths=["requests/adapters.py"],
            forbidden_file_paths=["tests/test_adapters.py"],
            max_latency_seconds=4.0,
        ),
        RetrievalScenario(
            name="Constructor follow-up",
            question="What is its constructor responsible for?",
            expected_chunk_names=["HTTPAdapter.__init__"],
            expected_file_paths=["requests/adapters.py"],
            forbidden_file_paths=["tests/test_adapters.py"],
            chat_history=[
                {"role": "user", "content": "Explain the HTTPAdapter class."},
                {
                    "role": "assistant",
                    "content": (
                        "The `HTTPAdapter` class provides transport adapter behavior.\n\n"
                        "Sources:\nrequests/adapters.py HTTPAdapter lines 167-240"
                    ),
                },
            ],
            max_latency_seconds=4.0,
        ),
        RetrievalScenario(
            name="Ambiguous implementation search",
            question="How are cookies handled?",
            expected_file_paths=["requests/cookies.py"],
            forbidden_file_paths=["tests/test_cookies.py"],
            max_latency_seconds=4.0,
        ),
        RetrievalScenario(
            name="Authentication implementation",
            question="Where is authentication implemented?",
            expected_file_paths=["requests/auth.py"],
            max_latency_seconds=4.0,
        ),
    ]

    print("=" * 80)
    print("RETRIEVAL REGRESSION CHECKS")
    print("=" * 80)
    print()

    for scenario in scenarios:
        print(f"[SCENARIO] {scenario.name}")
        print(f"Question: {scenario.question}")

        start_time = time.time()
        retrieval_query = scenario.question
        if scenario.chat_history:
            rag_chain = RAGChain.__new__(RAGChain)
            retrieval_query = rag_chain._build_retrieval_query(
                scenario.question,
                scenario.chat_history,
            )

        context = retriever.retrieve(
            query=retrieval_query,
            filters=RetrievalFilters(),
        )
        elapsed = time.time() - start_time

        print(
            f"Retrieved {len(context.results)} chunks "
            f"from {context.total_candidates} candidates "
            f"in {elapsed:.2f}s"
        )
        if scenario.chat_history:
            print("Rewritten retrieval query:")
            print(retrieval_query)

        if scenario.max_latency_seconds is not None:
            status = "PASS" if elapsed <= scenario.max_latency_seconds else "WARN"
            print(
                f"Latency check: {status} "
                f"(threshold {scenario.max_latency_seconds:.2f}s)"
            )

        top_results = context.results[:5]
        for index, result in enumerate(top_results, 1):
            print(
                f"  {index}. {result.file_path} :: {result.chunk_name} "
                f"[{result.chunk_type}] score={result.combined_score:.3f}"
            )

        _print_expectations(scenario, top_results)

        prompt = prompt_builder.build_prompt(
            question=scenario.question,
            retrieval_results=context.results,
        )
        print(f"Prompt size: {len(prompt)} chars")
        print()


def evaluate_query_rewriting():
    """Validate that follow-up questions are rewritten before retrieval."""
    try:
        RAGChain = _load_rag_chain()
    except Exception as exc:
        print("=" * 80)
        print("QUERY REWRITING UNAVAILABLE")
        print("=" * 80)
        print(f"Could not import RAGChain: {exc}")
        print()
        return

    fake_retriever = FakeRetriever()
    rag_chain = RAGChain.__new__(RAGChain)
    rag_chain.retriever = fake_retriever

    history = [
        {"role": "user", "content": "Explain the Session class in requests."},
        {
            "role": "assistant",
            "content": (
                "The `Session` class manages persistent settings.\n\n"
                "Sources:\nrequests/sessions.py Session lines 356-780"
            ),
        },
    ]

    retrieval_query = rag_chain._build_retrieval_query(
        "Which methods are most important and what does the first one do?",
        history,
    )

    print("=" * 80)
    print("FOLLOW-UP QUERY REWRITING")
    print("=" * 80)
    print(retrieval_query)
    print()


def evaluate_constructor_ranking_heuristics():
    """Validate constructor-specific reranking without external dependencies."""
    try:
        CodeRetriever = _load_code_retriever()
    except Exception as exc:
        print("=" * 80)
        print("CONSTRUCTOR HEURISTICS UNAVAILABLE")
        print("=" * 80)
        print(f"Could not import CodeRetriever: {exc}")
        print()
        return

    retriever = CodeRetriever.__new__(CodeRetriever)
    retriever.hybrid_alpha = 0.5
    retriever.rerank_top_k = 8
    retriever.min_similarity = 0.2

    constructor_result = retriever._create_result(
        chunk_id="ctor",
        text="def __init__(self, pool_connections=10): ...",
        metadata={
            "repository_name": "requests",
            "repository_id": "repo1",
            "file_path": "requests/adapters.py",
            "chunk_type": "method",
            "chunk_name": "HTTPAdapter.__init__",
            "class_name": "HTTPAdapter",
            "line_start": 194,
            "line_end": 224,
        },
        vector_score=0.45,
        bm25_score=0.8,
    )
    unrelated_method = retriever._create_result(
        chunk_id="headers",
        text="def add_headers(self, request): ...",
        metadata={
            "repository_name": "requests",
            "repository_id": "repo1",
            "file_path": "requests/adapters.py",
            "chunk_type": "method",
            "chunk_name": "HTTPAdapter.add_headers",
            "class_name": "HTTPAdapter",
            "line_start": 420,
            "line_end": 430,
        },
        vector_score=0.5,
        bm25_score=0.8,
    )
    class_header = retriever._create_result(
        chunk_id="class",
        text="class HTTPAdapter(BaseAdapter): ...",
        metadata={
            "repository_name": "requests",
            "repository_id": "repo1",
            "file_path": "requests/adapters.py",
            "chunk_type": "class",
            "chunk_name": "HTTPAdapter",
            "method_names": "HTTPAdapter.__init__,HTTPAdapter.send,HTTPAdapter.add_headers",
            "line_start": 167,
            "line_end": 470,
        },
        vector_score=0.44,
        bm25_score=0.75,
    )

    ranked = retriever._rerank_results(
        "What is its constructor responsible for in HTTPAdapter?",
        [unrelated_method, class_header, constructor_result],
    )

    print("=" * 80)
    print("CONSTRUCTOR HEURISTICS")
    print("=" * 80)
    for index, result in enumerate(ranked, 1):
        print(
            f"{index}. {result.chunk_name} [{result.chunk_type}] "
            f"score={result.rerank_score:.3f}"
        )
    print()


def _print_expectations(
    scenario: RetrievalScenario,
    results: List[object],
):
    chunk_names = [result.chunk_name or "" for result in results]
    file_paths = [result.file_path or "" for result in results]

    for expected in scenario.expected_chunk_names:
        status = "PASS" if any(expected in name for name in chunk_names) else "WARN"
        print(f"Expected chunk `{expected}`: {status}")

    for expected in scenario.expected_file_paths:
        status = "PASS" if any(expected in path for path in file_paths) else "WARN"
        print(f"Expected file `{expected}`: {status}")

    for forbidden in scenario.forbidden_file_paths:
        status = "PASS" if all(forbidden not in path for path in file_paths) else "WARN"
        print(f"Avoid file `{forbidden}`: {status}")


def _load_code_retriever():
    from app.rag.retriever import CodeRetriever
    return CodeRetriever


def _load_prompt_builder():
    from app.rag.prompt_builder import PromptBuilder
    return PromptBuilder


def _load_rag_chain():
    from app.rag.rag_chain import RAGChain
    return RAGChain


def _load_retrieval_filters():
    from app.rag.retrieval_models import RetrievalFilters
    return RetrievalFilters


def _load_retrieval_context():
    from app.rag.retrieval_models import RetrievalContext
    return RetrievalContext


if __name__ == "__main__":
    evaluate_query_rewriting()
    evaluate_constructor_ranking_heuristics()
    evaluate_live_retrieval()
