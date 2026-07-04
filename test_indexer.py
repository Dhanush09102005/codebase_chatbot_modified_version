#!/usr/bin/env python3
"""
Test script for the code indexer.

Demonstrates:
1. Single responsibility: ONLY persist chunks (no embedding, no retrieval)
2. Validates chunks have embeddings
3. Prepares consistent metadata
4. Incremental indexing (skip unchanged)
5. Batch inserts for efficiency
6. Comprehensive statistics
"""

from pathlib import Path
from app.ingestion import RepositoryManager, FileScanner
from app.rag import CodeParser, ASTChunker, CodeEmbedder, CodeIndexer


def test_indexer():
    """Test the code indexer pipeline."""
    
    print("=" * 80)
    print("CODE INDEXER TEST - Persistence Layer")
    print("=" * 80)
    print("\n🎯 SINGLE RESPONSIBILITY: Persist embedded chunks to vector store")
    print("   NOT responsible for: embedding, retrieval, or answer generation")
    print("   ONLY responsible for: storage with consistent metadata")
    print()
    
    # Initialize complete pipeline
    manager = RepositoryManager()
    scanner = FileScanner()
    parser = CodeParser()
    chunker = ASTChunker()
    embedder = CodeEmbedder()
    indexer = CodeIndexer()
    
    print("📊 Indexer Statistics:")
    print("-" * 80)
    stats = indexer.get_stats()
    for key, value in stats.items():
        print(f"  {key:20} {value}")
    print()
    
    # Clone repository
    repo_url = "https://github.com/fastapi/fastapi.git"
    
    print(f"📥 Loading repository: {repo_url}")
    repo_path = manager.load(repo_url)
    print(f"✅ Repository at: {repo_path}\n")
    
    # Scan Python files
    print("🔍 Scanning for code files...")
    code_files = scanner.scan(repo_path)
    python_files = [cf for cf in code_files if cf.language.value == "python"]
    print(f"✅ Found {len(python_files)} Python files\n")
    
    # Test complete pipeline
    if python_files:
        print("=" * 80)
        print("COMPLETE PIPELINE TEST")
        print("Scanner → Parser → Chunker → Embedder → Indexer → ChromaDB")
        print("=" * 80)
        
        # Process 5 files
        test_files = python_files[:5]
        
        print(f"\n⏳ Processing {len(test_files)} files through pipeline...")
        
        # Stage 1: Parse
        print("\n1️⃣  Parsing (extract AST)...")
        parsed = parser.parse_batch(test_files)
        print(f"   ✅ Parsed: {parsed}/{len(test_files)}")
        
        # Stage 2: Chunk
        print("\n2️⃣  Chunking (semantic units)...")
        chunked = chunker.chunk_batch(test_files)
        print(f"   ✅ Chunked: {chunked}/{len(test_files)}")
        
        # Stage 3: Embed
        print("\n3️⃣  Embedding (generate vectors)...")
        embedded = embedder.embed_files(test_files)
        print(f"   ✅ Embedded: {embedded}/{len(test_files)}")
        
        # Stage 4: Index (THE PERSISTENCE LAYER)
        print("\n4️⃣  Indexing (persist to ChromaDB)...")
        index_stats = indexer.index_files(test_files)
        
        print(f"\n📊 INDEXING STATISTICS:")
        print("-" * 80)
        stats_dict = index_stats.to_dict()
        for key, value in stats_dict.items():
            print(f"  {key:25} {value}")
        
        # Show what gets stored
        print(f"\n💾 WHAT GETS STORED IN CHROMADB:")
        print("-" * 80)
        
        if test_files[0].chunks:
            from app.rag.chunk_models import CodeChunk
            chunk = CodeChunk(**test_files[0].chunks[0])
            
            # Generate chunk ID (same as indexer)
            chunk_id = indexer._generate_chunk_id(chunk)
            print(f"\n  Chunk ID (stable, unique):")
            print(f"    {chunk_id}")
            
            print(f"\n  Embedding (vector):")
            print(f"    Dimension: {len(chunk.embedding)}")
            print(f"    First 5:   [{', '.join(f'{v:.4f}' for v in chunk.embedding[:5])}, ...]")
            
            print(f"\n  Document (text):")
            print(f"    {chunk.text[:100]}...")
            
            print(f"\n  Metadata (for filtering):")
            metadata = indexer._prepare_metadata(chunk)
            for key, value in sorted(metadata.items()):
                val_str = str(value)[:50]
                if len(str(value)) > 50:
                    val_str += "..."
                print(f"    {key:20} {val_str}")
        
        # Test incremental indexing
        print(f"\n" + "=" * 80)
        print("INCREMENTAL INDEXING TEST (Only Changed Chunks)")
        print("=" * 80)
        
        print(f"\n⏳ Re-indexing same files (should skip existing)...")
        index_stats2 = indexer.index_files(test_files)
        
        print(f"\n📊 SECOND INDEXING (Incremental):")
        print("-" * 80)
        print(f"  Total chunks:     {index_stats2.total_chunks}")
        print(f"  Indexed:          {index_stats2.indexed_chunks}")
        print(f"  Skipped:          {index_stats2.skipped_chunks} ← Already exist!")
        print(f"  Failed:           {index_stats2.failed_chunks}")
        print(f"  Time:             {index_stats2.indexing_time_seconds:.2f}s")
        
        print(f"\n💡 This is incremental indexing!")
        print(f"   Only changed chunks get re-indexed.")
        print(f"   Unchanged chunks are skipped (efficient!).")
        
        # Show vector store stats
        print(f"\n" + "=" * 80)
        print("VECTOR STORE STATISTICS")
        print("=" * 80)
        
        print(f"\n  Total indexed chunks: {indexer.vector_store.count()}")
        
        if hasattr(indexer.vector_store, 'get_collection_info'):
            collection_info = indexer.vector_store.get_collection_info()
            for key, value in collection_info.items():
                print(f"  {key:20} {value}")
        
        # Show why single responsibility matters
        print(f"\n" + "=" * 80)
        print("WHY SINGLE RESPONSIBILITY MATTERS")
        print("=" * 80)
        
        print(f"\n  ✅ CodeIndexer ONLY does:")
        print(f"     1. Validates chunks have embeddings")
        print(f"     2. Generates stable chunk IDs")
        print(f"     3. Prepares consistent metadata")
        print(f"     4. Checks for existing chunks (incremental)")
        print(f"     5. Batch inserts to vector store")
        print(f"     6. Returns statistics")
        
        print(f"\n  ❌ CodeIndexer does NOT:")
        print(f"     1. Generate embeddings (that's CodeEmbedder)")
        print(f"     2. Retrieve chunks (that's Retriever)")
        print(f"     3. Answer questions (that's LLM Chain)")
        print(f"     4. Parse code (that's CodeParser)")
        print(f"     5. Chunk code (that's ASTChunker)")
        
        print(f"\n  💡 Benefits:")
        print(f"     - Easy to test (focused responsibility)")
        print(f"     - Easy to swap vector stores (interface)")
        print(f"     - Easy to explain in interviews")
        print(f"     - Clean architecture")
        
        # Show vector store abstraction
        print(f"\n" + "=" * 80)
        print("VECTOR STORE ABSTRACTION (Easy to Swap)")
        print("=" * 80)
        
        print(f"\n  Current: ChromaVectorStore")
        print(f"\n  Can easily swap to:")
        print(f"     - MilvusVectorStore")
        print(f"     - QdrantVectorStore")
        print(f"     - PineconeVectorStore")
        print(f"     - WeaviateVectorStore")
        print(f"\n  Just implement VectorStore interface:")
        print(f"     - add_documents()")
        print(f"     - document_exists()")
        print(f"     - delete_documents()")
        print(f"     - count()")
        print(f"     - clear()")
        
        # Show complete pipeline
        print(f"\n" + "=" * 80)
        print("COMPLETE PIPELINE ACHIEVED")
        print("=" * 80)
        
        print(f"\n  RepositoryManager → repo_path")
        print(f"       ↓")
        print(f"  FileScanner      → CodeFile (basic metadata)")
        print(f"       ↓")
        print(f"  Parser           → CodeFile (+ AST, functions, classes)  ✅")
        print(f"       ↓")
        print(f"  Chunker          → CodeFile (+ semantic chunks)          ✅")
        print(f"       ↓")
        print(f"  Embedder         → CodeFile (+ embeddings)               ✅")
        print(f"       ↓")
        print(f"  Indexer          → ChromaDB (persisted!)                 ✅ DONE!")
        print(f"       ↓")
        print(f"  Retriever        → CodeFile (+ scores)                   ⏳ NEXT")
        print(f"       ↓")
        print(f"  LLM Chain        → Answer                                ⏳ NEXT")
        
        print(f"\n  🎯 Pipeline is almost complete!")
        print(f"     Just need: Hybrid Retriever + LLM Chain")
    
    print("\n" + "=" * 80)
    print("✅ INDEXER TEST COMPLETE")
    print("=" * 80)
    
    print(f"\n💡 Interview Talking Point:")
    print(f"   'The indexer has a single responsibility: persist embedded chunks.'")
    print(f"   'It validates, prepares metadata, handles incremental updates,'")
    print(f"   'and batch inserts efficiently. It's NOT an embedder or retriever.'")
    print(f"   'This separation makes it testable, swappable, and maintainable.'")


if __name__ == "__main__":
    test_indexer()
