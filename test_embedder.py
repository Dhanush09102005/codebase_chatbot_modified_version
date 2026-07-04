#!/usr/bin/env python3
"""
Test script for the embedding pipeline.

Demonstrates:
1. Embedding pipeline (not just model.encode())
2. Works with CodeChunk objects (not raw strings)
3. Progressive enrichment (enriches same objects)
4. Batch processing for efficiency
5. Metadata preparation for better embeddings
"""

from pathlib import Path
from app.ingestion import RepositoryManager, FileScanner
from app.rag import CodeParser, ASTChunker, CodeEmbedder


def test_embedder():
    """Test the embedding pipeline on real code."""
    
    print("=" * 80)
    print("EMBEDDING PIPELINE TEST")
    print("=" * 80)
    print("\n💡 This is a PIPELINE, not just model.encode()!")
    print("   - Accepts CodeChunk objects (not raw strings)")
    print("   - Prepares text with metadata for better embeddings")
    print("   - Batches efficiently")
    print("   - Enriches same objects (progressive enrichment)")
    print()
    
    # Initialize pipeline components
    manager = RepositoryManager()
    scanner = FileScanner()
    parser = CodeParser()
    chunker = ASTChunker()
    embedder = CodeEmbedder()
    
    print(f"📊 Embedder Statistics:")
    print("-" * 80)
    stats = embedder.get_stats()
    for key, value in stats.items():
        print(f"  {key:20} {value}")
    print()
    
    # Clone repository
    repo_url = "https://github.com/fastapi/fastapi.git"
    
    print(f"📥 Loading repository: {repo_url}")
    repo_path = manager.load(repo_url)
    print(f"✅ Repository at: {repo_path}\n")
    
    # Scan for Python files
    print("🔍 Scanning for code files...")
    code_files = scanner.scan(repo_path)
    python_files = [cf for cf in code_files if cf.language.value == "python"]
    print(f"✅ Found {len(python_files)} Python files\n")
    
    # Test complete pipeline on one file
    if python_files:
        print("=" * 80)
        print("COMPLETE PIPELINE TEST (Scanner → Parser → Chunker → Embedder)")
        print("=" * 80)
        
        sample_file = python_files[0]
        
        print(f"\n📄 File: {sample_file.relative_path_str}")
        print(f"   Size: {sample_file.size} bytes")
        
        # Stage 1: FileScanner (already done)
        print(f"\n✅ Stage 1: FileScanner")
        print(f"   Created CodeFile with basic metadata")
        
        # Stage 2: Parser
        print(f"\n⏳ Stage 2: Parser - Extracting AST...")
        parser.parse(sample_file)
        print(f"✅ Parsed! Found {len(sample_file.functions or [])} functions, {len(sample_file.classes or [])} classes")
        
        # Stage 3: Chunker
        print(f"\n⏳ Stage 3: Chunker - Creating semantic chunks...")
        chunker.chunk(sample_file)
        print(f"✅ Chunked! Created {len(sample_file.chunks or [])} semantic units")
        
        # Stage 4: Embedder
        print(f"\n⏳ Stage 4: Embedder - Generating embeddings...")
        print(f"   Before: is_embedded() = {sample_file.is_embedded()}")
        print(f"   Before: embeddings = {sample_file.embeddings}")
        
        success = embedder.embed_file(sample_file)
        
        if success:
            print(f"✅ Embedded!")
            print(f"   After: is_embedded() = {sample_file.is_embedded()}")
            print(f"   After: embeddings = {len(sample_file.embeddings or [])} vectors")
            print(f"   Embedding dim: {len(sample_file.embeddings[0])} (all-MiniLM-L6-v2 = 384)")
            
            # Show embedding sample
            if sample_file.embeddings:
                print(f"\n📊 SAMPLE EMBEDDING:")
                print("-" * 80)
                embedding = sample_file.embeddings[0]
                print(f"  Vector length: {len(embedding)}")
                print(f"  First 10 dims: {embedding[:10]}")
                print(f"  Last 10 dims:  {embedding[-10:]}")
                
                # Check normalization
                import numpy as np
                norm = np.linalg.norm(embedding)
                print(f"  L2 norm:       {norm:.6f} (should be ~1.0 if normalized)")
            
            # Show chunks with embeddings
            print(f"\n📦 CHUNKS WITH EMBEDDINGS:")
            print("-" * 80)
            for i, chunk_dict in enumerate(sample_file.chunks[:5], 1):
                print(f"\n  Chunk {i}:")
                print(f"    Type:       {chunk_dict['chunk_type']}")
                print(f"    Name:       {chunk_dict['name']}")
                print(f"    Lines:      {chunk_dict['line_start']}-{chunk_dict['line_end']}")
                print(f"    Text size:  {chunk_dict['char_count']} chars")
                if 'embedding' in chunk_dict and chunk_dict['embedding']:
                    print(f"    Embedding:  ✅ {len(chunk_dict['embedding'])}-dim vector")
                    # Show first few dimensions
                    emb = chunk_dict['embedding'][:5]
                    print(f"    First 5:    [{', '.join(f'{v:.4f}' for v in emb)}, ...]")
                else:
                    print(f"    Embedding:  ❌ None")
            
            if len(sample_file.chunks) > 5:
                print(f"\n  ... and {len(sample_file.chunks) - 5} more chunks with embeddings")
        
        # Test metadata preparation
        print(f"\n" + "=" * 80)
        print("METADATA PREPARATION (Better Embeddings)")
        print("=" * 80)
        
        if sample_file.chunks:
            from app.rag.chunk_models import CodeChunk
            chunk = CodeChunk(**sample_file.chunks[0])
            
            print(f"\n📝 Original chunk text (first 200 chars):")
            print("-" * 80)
            print(chunk.text[:200] + "...")
            
            print(f"\n✨ Prepared text with metadata (what actually gets embedded):")
            print("-" * 80)
            prepared = embedder._prepare_text(chunk)
            print(prepared[:400] + "...")
            
            print(f"\n💡 Why this matters:")
            print("   Metadata helps the embedding model understand:")
            print("   - What type of code (function, class, method)")
            print("   - What it's named")
            print("   - What file it's from")
            print("   - What it does (docstring)")
            print("   This leads to better semantic search!")
        
        # Test batch embedding
        print(f"\n" + "=" * 80)
        print("BATCH EMBEDDING TEST (Efficiency)")
        print("=" * 80)
        
        # Process 5 more files
        batch_files = python_files[1:6]
        print(f"\n⏳ Processing {len(batch_files)} files...")
        
        # Parse
        parsed = parser.parse_batch(batch_files)
        print(f"✅ Parsed: {parsed}/{len(batch_files)}")
        
        # Chunk
        chunked = chunker.chunk_batch(batch_files)
        print(f"✅ Chunked: {chunked}/{len(batch_files)}")
        
        # Embed (THE KEY PART)
        print(f"\n⏳ Embedding {len(batch_files)} files...")
        embedded = embedder.embed_files(batch_files)
        print(f"✅ Embedded: {embedded}/{len(batch_files)}")
        
        # Statistics
        print(f"\n📊 BATCH STATISTICS:")
        print("-" * 80)
        
        total_chunks = sum(len(cf.chunks or []) for cf in batch_files if cf.is_chunked())
        total_embeddings = sum(len(cf.embeddings or []) for cf in batch_files if cf.is_embedded())
        
        print(f"  Total chunks:      {total_chunks}")
        print(f"  Total embeddings:  {total_embeddings}")
        print(f"  Embedding dim:     {embedder.embedding_dim}")
        print(f"  Model:             {embedder.model.model_name}")
        
        # Show progressive enrichment
        print(f"\n" + "=" * 80)
        print("PROGRESSIVE ENRICHMENT COMPLETE")
        print("=" * 80)
        
        if batch_files:
            cf = batch_files[0]
            print(f"\n📄 File: {cf.relative_path_str}")
            print(f"\n  Stage 1 - FileScanner:")
            print(f"    ✅ path, language, size, content_hash, repository_name")
            print(f"\n  Stage 2 - Parser:")
            print(f"    ✅ + imports ({len(cf.imports or [])}), functions ({len(cf.functions or [])}), classes ({len(cf.classes or [])})")
            print(f"\n  Stage 3 - Chunker:")
            print(f"    ✅ + chunks ({len(cf.chunks or [])} semantic units)")
            print(f"\n  Stage 4 - Embedder:")
            print(f"    ✅ + embeddings ({len(cf.embeddings or [])} vectors, {embedder.embedding_dim}-dim)")
            print(f"\n  Stage 5 - Indexer (next):")
            print(f"    ⏳ Store in ChromaDB for retrieval")
            print(f"\n  Stage 6 - Retriever (next):")
            print(f"    ⏳ + metadata (bm25_score, vector_score, combined_score)")
            print()
            print(f"  💡 Same CodeFile object through entire pipeline!")
        
        # Why this is NOT just model.encode()
        print(f"\n" + "=" * 80)
        print("WHY THIS IS A PIPELINE, NOT JUST model.encode()")
        print("=" * 80)
        
        print(f"\n  ❌ Simple wrapper:")
        print(f"     def embed(text: str) -> List[float]:")
        print(f"         return model.encode(text)")
        
        print(f"\n  ✅ Our pipeline:")
        print(f"     1. Accepts CodeChunk objects (not strings)")
        print(f"     2. Prepares text with metadata (type, name, file, docstring)")
        print(f"     3. Batches efficiently")
        print(f"     4. Enriches same objects (progressive enrichment)")
        print(f"     5. Works with CodeFile objects")
        print(f"     6. Handles errors gracefully")
        print(f"     7. Supports multiple embedding models")
        print(f"     8. Can be extended with caching, filtering, etc.")
        
        print(f"\n  This architectural discipline makes everything fit together! 🎯")
    
    print("\n" + "=" * 80)
    print("✅ EMBEDDER TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_embedder()
