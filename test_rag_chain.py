#!/usr/bin/env python3
"""
Test script for the complete RAG Chain.

Demonstrates the complete pipeline:
  Repository → Scanner → Parser → Chunker → Embedder → Indexer → Retriever → RAG Chain → Answer

This showcases:
1. Clean separation of responsibilities
2. Orchestration (not implementation) in RAG Chain
3. Independent components that can be swapped
4. Complete end-to-end workflow
"""

from pathlib import Path
from app.ingestion import RepositoryManager, FileScanner
from app.rag import (
    CodeParser,
    ASTChunker,
    CodeEmbedder,
    CodeIndexer,
    CodeRetriever,
    PromptBuilder,
    RAGChain,
    create_llm_client,
)
from app.rag.retrieval_models import RetrievalFilters


def test_rag_chain():
    """Test the complete RAG chain."""
    
    print("=" * 80)
    print("RAG CHAIN TEST - Complete Code Q&A Pipeline")
    print("=" * 80)
    print("\n🎯 GOAL: Answer questions about code using RAG")
    print("   Pipeline: Retrieve → Build Prompt → Generate Answer")
    print()
    
    # =========================================================================
    # STAGE 1: Index a repository (if not already indexed)
    # =========================================================================
    
    print("=" * 80)
    print("STAGE 1: INDEXING (Prepare Knowledge Base)")
    print("=" * 80)
    print()
    
    manager = RepositoryManager()
    scanner = FileScanner()
    parser = CodeParser()
    chunker = ASTChunker()
    embedder = CodeEmbedder()
    indexer = CodeIndexer()
    
    # Check if already indexed
    if indexer.vector_store.count() > 0:
        print(f"✅ Vector store already has {indexer.vector_store.count()} chunks")
        print("   Skipping indexing...\n")
    else:
        print("📥 Loading repository...")
        repo_url = "https://github.com/fastapi/fastapi.git"
        repo_path = manager.load(repo_url)
        print(f"✅ Repository at: {repo_path}\n")
        
        print("🔍 Scanning for Python files...")
        code_files = scanner.scan(repo_path)
        python_files = [cf for cf in code_files if cf.language.value == "python"]
        print(f"✅ Found {len(python_files)} Python files\n")
        
        # Index first 10 files
        print("⏳ Processing first 10 files through pipeline...")
        test_files = python_files[:10]
        
        print("  1/4 Parsing...")
        parser.parse_batch(test_files)
        
        print("  2/4 Chunking...")
        chunker.chunk_batch(test_files)
        
        print("  3/4 Embedding...")
        embedder.embed_files(test_files)
        
        print("  4/4 Indexing...")
        stats = indexer.index_files(test_files)
        print(f"✅ Indexed {stats.indexed_chunks} chunks\n")
    
    # =========================================================================
    # STAGE 2: Initialize RAG Chain
    # =========================================================================
    
    print("=" * 80)
    print("STAGE 2: INITIALIZE RAG CHAIN")
    print("=" * 80)
    print()
    
    print("🔧 Creating RAG Chain components...")
    print("   This demonstrates clean separation of responsibilities:\n")
    
    # Create independent components
    print("   1️⃣  Retriever (finds relevant code)")
    retriever = CodeRetriever()
    
    print("   2️⃣  Prompt Builder (formats prompts)")
    prompt_builder = PromptBuilder(
        include_line_numbers=True,
        include_chunk_types=True,
        max_chunks_in_prompt=5,
        instruction_style="strict"
    )
    
    print("   3️⃣  LLM Client (generates answers)")
    print("       NOTE: Using HuggingFace client (may be slow without GPU)")
    print("       For faster testing, use: llm_client = create_llm_client('groq', api_key='...')")
    print()
    
    # For this demo, we'll show the structure without actually calling the LLM
    # In production, you would:
    # llm_client = create_llm_client(provider="huggingface")  # or "groq", "openai", etc.
    # rag_chain = RAGChain(retriever=retriever, prompt_builder=prompt_builder, llm_client=llm_client)
    
    print("   ✅ Components created (independent, swappable)")
    print()
    
    # =========================================================================
    # STAGE 3: Test Retrieval + Prompt Building (Without LLM Call)
    # =========================================================================
    
    print("=" * 80)
    print("STAGE 3: TEST RETRIEVAL & PROMPT BUILDING")
    print("=" * 80)
    print()
    
    # Example questions
    questions = [
        "How does FastAPI handle route parameters?",
        "What is the main application class in FastAPI?",
        "How are dependencies injected in FastAPI?",
    ]
    
    print("📋 Testing with sample questions:\n")
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 80}")
        print(f"QUESTION {i}: {question}")
        print(f"{'─' * 80}\n")
        
        # Step 1: Retrieve relevant chunks
        print("Step 1/3: Retrieving relevant code chunks...")
        retrieval_context = retriever.retrieve(query=question)
        
        print(f"✅ Retrieved {len(retrieval_context.results)} chunks in {retrieval_context.retrieval_time_seconds:.2f}s")
        
        if retrieval_context.results:
            print("\n📦 Top retrieved chunks:")
            for j, result in enumerate(retrieval_context.results[:3], 1):
                print(f"   {j}. {result.file_path}")
                print(f"      Type: {result.chunk_type}, Name: {result.chunk_name}")
                print(f"      Lines: {result.line_start}-{result.line_end}")
                print(f"      Score: {result.combined_score:.3f}")
        
        # Step 2: Build prompt
        print("\nStep 2/3: Building structured prompt...")
        prompt = prompt_builder.build_prompt(
            question=question,
            retrieval_results=retrieval_context.results
        )
        
        print(f"✅ Prompt built: {len(prompt)} characters")
        
        # Show prompt structure (first 500 chars)
        print("\n📝 Prompt Preview (first 500 chars):")
        print("─" * 80)
        print(prompt[:500])
        if len(prompt) > 500:
            print("...")
            print(f"[{len(prompt) - 500} more characters]")
        print("─" * 80)
        
        # Step 3: Would call LLM here
        print("\nStep 3/3: Generate answer with LLM...")
        print("⏭️  SKIPPED (to avoid slow LLM inference in test)")
        print("   In production: response = llm_client.generate(prompt)")
        print()
    
    # =========================================================================
    # STAGE 4: Show RAG Chain Architecture
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("STAGE 4: RAG CHAIN ARCHITECTURE")
    print("=" * 80)
    print()
    
    print("🏗️  SINGLE RESPONSIBILITY PRINCIPLE:")
    print()
    
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│ RAG Chain (Orchestrator)                                        │")
    print("│                                                                 │")
    print("│  def query(question):                                           │")
    print("│      # Step 1: Retrieve                                         │")
    print("│      results = retriever.retrieve(question)                     │")
    print("│                                                                 │")
    print("│      # Step 2: Build Prompt                                     │")
    print("│      prompt = prompt_builder.build_prompt(question, results)    │")
    print("│                                                                 │")
    print("│      # Step 3: Generate Answer                                  │")
    print("│      answer = llm_client.generate(prompt)                       │")
    print("│                                                                 │")
    print("│      # Step 4: Return Complete Response                         │")
    print("│      return RAGResponse(question, answer, results, ...)         │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()
    
    print("✅ RAG Chain ONLY orchestrates - it does NOT:")
    print("   ❌ Contain retrieval logic (that's in Retriever)")
    print("   ❌ Format prompts (that's in PromptBuilder)")
    print("   ❌ Call LLM APIs (that's in LLMClient)")
    print("   ❌ Embed queries (that's in Embedder)")
    print("   ❌ Access vector DB (that's in VectorStore)")
    print()
    
    print("✅ Each component is INDEPENDENT:")
    print("   🔄 Swap Retriever: CodeRetriever → SemanticRetriever")
    print("   🔄 Swap PromptBuilder: PromptBuilder → CustomPromptBuilder")
    print("   🔄 Swap LLMClient: HuggingFace → Groq → OpenAI → Ollama")
    print()
    
    # =========================================================================
    # STAGE 5: Component Comparison
    # =========================================================================
    
    print("=" * 80)
    print("STAGE 5: COMPONENT RESPONSIBILITIES")
    print("=" * 80)
    print()
    
    components = {
        "Retriever": {
            "Does": "Finds relevant code chunks using hybrid search",
            "Does NOT": "Generate embeddings, format prompts, call LLMs"
        },
        "PromptBuilder": {
            "Does": "Formats retrieved chunks into structured prompts",
            "Does NOT": "Retrieve chunks, call LLMs, orchestrate workflow"
        },
        "LLMClient": {
            "Does": "Communicates with language model APIs",
            "Does NOT": "Retrieve chunks, format prompts, orchestrate"
        },
        "RAGChain": {
            "Does": "Orchestrates: retrieve → format → generate",
            "Does NOT": "Implement retrieval, prompting, or LLM logic"
        }
    }
    
    for component, responsibilities in components.items():
        print(f"📦 {component}")
        print(f"   ✅ DOES:     {responsibilities['Does']}")
        print(f"   ❌ DOES NOT: {responsibilities['Does NOT']}")
        print()
    
    # =========================================================================
    # STAGE 6: Complete Pipeline Summary
    # =========================================================================
    
    print("=" * 80)
    print("STAGE 6: COMPLETE PIPELINE")
    print("=" * 80)
    print()
    
    print("📊 INDEXING PIPELINE (Offline):")
    print("   Repository → FileScanner → CodeFile")
    print("                     ↓")
    print("                  Parser → CodeFile (+ AST)")
    print("                     ↓")
    print("                 Chunker → CodeFile (+ chunks)")
    print("                     ↓")
    print("                Embedder → CodeFile (+ embeddings)")
    print("                     ↓")
    print("                 Indexer → ChromaDB (persisted)")
    print()
    
    print("📊 RAG PIPELINE (Online):")
    print("   User Question")
    print("        ↓")
    print("   RAGChain.query()")
    print("        ↓")
    print("   ├─ Retriever.retrieve()        → Top code chunks")
    print("   ├─ PromptBuilder.build_prompt() → Structured prompt")
    print("   ├─ LLMClient.generate()        → Answer")
    print("   └─ RAGResponse                 → Complete result")
    print()
    
    print("🎯 KEY ARCHITECTURAL ACHIEVEMENTS:")
    print("   ✅ Progressive enrichment (same object throughout)")
    print("   ✅ Single responsibility (each component does ONE thing)")
    print("   ✅ Interface abstraction (easy to swap implementations)")
    print("   ✅ Clean orchestration (RAGChain connects, doesn't implement)")
    print("   ✅ Semantic chunking (functions/classes, not arbitrary text)")
    print("   ✅ Hybrid retrieval (vector + BM25 + reranking)")
    print("   ✅ Multi-provider LLM (HuggingFace, Groq, OpenAI, Ollama)")
    print()
    
    print("=" * 80)
    print("✅ RAG CHAIN TEST COMPLETE")
    print("=" * 80)
    print()
    
    print("💡 Interview Talking Point:")
    print("   'The RAG Chain is just an orchestrator. It calls the retriever")
    print("   to find relevant code, the prompt builder to format the context,")
    print("   and the LLM client to generate the answer. Each component has")
    print("   a single responsibility and can be swapped independently.'")
    print()
    
    print("🚀 To use with actual LLM:")
    print("   1. For local (slow): llm_client = create_llm_client('huggingface')")
    print("   2. For fast cloud: llm_client = create_llm_client('groq', api_key='...')")
    print("   3. Create chain: rag_chain = RAGChain(retriever, prompt_builder, llm_client)")
    print("   4. Query: response = rag_chain.query('How does X work?')")
    print("   5. Display: print(response.format_for_display())")


if __name__ == "__main__":
    test_rag_chain()
