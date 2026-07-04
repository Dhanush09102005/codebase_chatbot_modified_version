#!/usr/bin/env python3
"""
Test script for AST-based semantic chunking.

Demonstrates the KEY DIFFERENTIATOR:
- Traditional RAG: Split text into 512-char chunks (arbitrary)
- Our approach: Chunk by functions/classes (semantic)
"""

from pathlib import Path
from app.ingestion import RepositoryManager, FileScanner
from app.rag import CodeParser, ASTChunker, CodeChunk


def test_chunker():
    """Test the AST-based chunker on a real repository."""
    
    print("=" * 80)
    print("AST-BASED SEMANTIC CHUNKING TEST")
    print("=" * 80)
    print("\n🎯 KEY DIFFERENTIATOR:")
    print("   ❌ Traditional RAG: characters 0-512, 513-1024 (arbitrary)")
    print("   ✅ Our approach: function(), class{} (semantic units)")
    print()
    
    # Initialize components
    manager = RepositoryManager()
    scanner = FileScanner()
    parser = CodeParser()
    chunker = ASTChunker()
    
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
    
    # Test the complete pipeline on one file
    if python_files:
        print("=" * 80)
        print("COMPLETE PIPELINE TEST (FileScanner → Parser → Chunker)")
        print("=" * 80)
        
        sample_file = python_files[0]
        
        print(f"\n📄 File: {sample_file.relative_path_str}")
        print(f"   Size: {sample_file.size} bytes")
        print(f"   Language: {sample_file.language.value}")
        
        # Stage 1: FileScanner (already done)
        print(f"\n✅ Stage 1: FileScanner")
        print(f"   Created CodeFile with basic metadata")
        print(f"   - repository_name: {sample_file.repository_name}")
        print(f"   - content_hash: {sample_file.content_hash[:16]}...")
        print(f"   - language: {sample_file.language.value}")
        
        # Stage 2: Parser
        print(f"\n⏳ Stage 2: Parser - Extracting AST...")
        parser.parse(sample_file)
        print(f"✅ Parsed! Enriched SAME CodeFile with:")
        print(f"   - functions: {len(sample_file.functions or [])} found")
        print(f"   - classes: {len(sample_file.classes or [])} found")
        print(f"   - imports: {len(sample_file.imports or [])} found")
        
        # Stage 3: Chunker
        print(f"\n⏳ Stage 3: Chunker - Creating semantic chunks...")
        chunker.chunk(sample_file)
        print(f"✅ Chunked! Enriched SAME CodeFile with:")
        print(f"   - chunks: {len(sample_file.chunks or [])} semantic units")
        
        # Show the chunks
        if sample_file.chunks:
            print(f"\n📦 CHUNKS (Semantic Units):")
            print("-" * 80)
            
            for i, chunk_dict in enumerate(sample_file.chunks[:10], 1):
                print(f"\n  Chunk {i}: {chunk_dict['chunk_type'].upper()}")
                print(f"    Name:       {chunk_dict['name']}")
                print(f"    File:       {chunk_dict['file_path']}")
                print(f"    Lines:      {chunk_dict['line_start']}-{chunk_dict['line_end']} ({chunk_dict['line_count']} lines)")
                print(f"    Size:       {chunk_dict['char_count']} chars")
                print(f"    Language:   {chunk_dict['language']}")
                
                # Show metadata
                if chunk_dict.get('metadata'):
                    meta = chunk_dict['metadata']
                    if meta.get('params'):
                        print(f"    Params:     {meta['params']}")
                    if meta.get('decorators'):
                        print(f"    Decorators: {', '.join(meta['decorators'])}")
                    if meta.get('docstring'):
                        doc = meta['docstring'][:50] + "..." if len(meta['docstring']) > 50 else meta['docstring']
                        print(f"    Docstring:  {doc}")
                    if meta.get('method_names'):
                        print(f"    Methods:    {', '.join(meta['method_names'][:3])}")
                
                # Show snippet of text
                text_snippet = chunk_dict['text'][:100].replace('\n', ' ')
                if len(chunk_dict['text']) > 100:
                    text_snippet += "..."
                print(f"    Preview:    {text_snippet}")
            
            if len(sample_file.chunks) > 10:
                print(f"\n  ... and {len(sample_file.chunks) - 10} more chunks")
        
        # Show chunk type distribution
        print(f"\n📊 CHUNK TYPE DISTRIBUTION:")
        print("-" * 80)
        chunk_types = {}
        for chunk_dict in sample_file.chunks:
            chunk_type = chunk_dict['chunk_type']
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        for chunk_type, count in sorted(chunk_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {chunk_type:15} {count:4} chunks")
        
        # Comparison with traditional RAG
        print(f"\n🔍 TRADITIONAL RAG vs OUR APPROACH:")
        print("-" * 80)
        
        # Calculate what traditional RAG would do
        traditional_chunk_size = 512
        traditional_chunks = (sample_file.size // traditional_chunk_size) + 1
        
        print(f"\n  ❌ Traditional RAG (text splitting):")
        print(f"     Would create ~{traditional_chunks} chunks of 512 chars each")
        print(f"     Problem: Might split mid-function, loses context")
        print(f"     Example chunks:")
        print(f"       - Chunk 1: characters 0-512 (arbitrary)")
        print(f"       - Chunk 2: characters 513-1024 (arbitrary)")
        print(f"       - Chunk 3: characters 1025-1536 (arbitrary)")
        
        print(f"\n  ✅ Our Approach (AST-aware):")
        print(f"     Created {len(sample_file.chunks)} semantic chunks")
        print(f"     Benefit: Each chunk is a complete function/class")
        print(f"     Example chunks:")
        if sample_file.chunks:
            for i, chunk_dict in enumerate(sample_file.chunks[:3], 1):
                print(f"       - Chunk {i}: {chunk_dict['chunk_type']} '{chunk_dict['name']}' (lines {chunk_dict['line_start']}-{chunk_dict['line_end']})")
        
        # Show why this matters for retrieval
        print(f"\n💡 WHY THIS MATTERS FOR RETRIEVAL:")
        print("-" * 80)
        print("  When user asks: 'How do I create an access token?'")
        print()
        print("  ❌ Traditional RAG might return:")
        print("     - Random text chunk that mentions 'token'")
        print("     - Incomplete function (split across chunks)")
        print("     - No context about what file/line it's from")
        print()
        print("  ✅ Our approach returns:")
        print("     - Complete create_access_token() function")
        print("     - With decorators, parameters, docstring")
        print("     - Exact file path and line numbers")
        print("     - Can jump directly to definition")
        print()
        print("  This is the interview talking point! 🎯")
        
        # Test batch chunking
        print(f"\n" + "=" * 80)
        print("BATCH PROCESSING TEST")
        print("=" * 80)
        
        # Parse and chunk 10 files
        batch_files = python_files[:10]
        print(f"\n⏳ Processing {len(batch_files)} files...")
        
        # Parse batch
        parsed_count = parser.parse_batch(batch_files)
        print(f"✅ Parsed: {parsed_count}/{len(batch_files)} files")
        
        # Chunk batch
        chunked_count = chunker.chunk_batch(batch_files)
        print(f"✅ Chunked: {chunked_count}/{len(batch_files)} files")
        
        # Statistics
        print(f"\n📊 BATCH STATISTICS:")
        print("-" * 80)
        
        total_chunks = sum(len(cf.chunks or []) for cf in batch_files if cf.is_chunked())
        total_functions = sum(len(cf.functions or []) for cf in batch_files if cf.is_parsed())
        total_classes = sum(len(cf.classes or []) for cf in batch_files if cf.is_parsed())
        
        print(f"  Total chunks:     {total_chunks}")
        print(f"  Total functions:  {total_functions}")
        print(f"  Total classes:    {total_classes}")
        print(f"  Avg chunks/file:  {total_chunks / len(batch_files):.1f}")
        
        # Show progressive enrichment
        print(f"\n" + "=" * 80)
        print("PROGRESSIVE ENRICHMENT PATTERN")
        print("=" * 80)
        
        if batch_files:
            cf = batch_files[0]
            print(f"\n📄 File: {cf.relative_path_str}")
            print(f"\n  Stage 1 - FileScanner:")
            print(f"    ✅ absolute_path, relative_path, language, size, content_hash")
            print(f"\n  Stage 2 - Parser:")
            print(f"    ✅ + imports ({len(cf.imports or [])}), functions ({len(cf.functions or [])}), classes ({len(cf.classes or [])}), ast_tree")
            print(f"\n  Stage 3 - Chunker:")
            print(f"    ✅ + chunks ({len(cf.chunks or [])} semantic units)")
            print(f"\n  Stage 4 - Embedder (next):")
            print(f"    ⏳ + embeddings (vector for each chunk)")
            print(f"\n  Stage 5 - Retriever (next):")
            print(f"    ⏳ + metadata (bm25_score, vector_score, combined_score)")
            print()
            print(f"  💡 Same CodeFile object flows through entire pipeline!")
            print(f"     No data loss, easy debugging, efficient caching!")
    
    print("\n" + "=" * 80)
    print("✅ CHUNKER TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_chunker()
