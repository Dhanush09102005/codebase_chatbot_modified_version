#!/usr/bin/env python3
"""
Test script for the AST-based code parser.

Demonstrates:
1. Parsing Python files with tree-sitter
2. Extracting functions, classes, imports, docstrings
3. Progressive enrichment pattern (CodeFile stays the same)
4. Line numbers for jump-to-definition
"""

from pathlib import Path
from app.ingestion import RepositoryManager, FileScanner
from app.rag import CodeParser


def test_parser():
    """Test the code parser on a real repository."""
    
    print("=" * 80)
    print("CODE PARSER TEST - AST-Based Structure Extraction")
    print("=" * 80)
    
    # Initialize components
    manager = RepositoryManager()
    scanner = FileScanner()
    parser = CodeParser()
    
    # Clone a small Python repository
    repo_url = "https://github.com/fastapi/fastapi.git"
    
    print(f"\n📥 Loading repository: {repo_url}")
    repo_path = manager.load(repo_url)
    print(f"✅ Repository at: {repo_path}")
    
    # Scan for Python files
    print(f"\n🔍 Scanning for code files...")
    code_files = scanner.scan(repo_path)
    
    # Filter Python files for testing
    python_files = [cf for cf in code_files if cf.language.value == "python"]
    print(f"✅ Found {len(python_files)} Python files")
    
    # Parse a sample file
    if python_files:
        print("\n" + "=" * 80)
        print("PARSING SAMPLE FILE")
        print("=" * 80)
        
        # Take first Python file
        sample_file = python_files[0]
        
        print(f"\n📄 File: {sample_file.relative_path_str}")
        print(f"   Language: {sample_file.language.value}")
        print(f"   Size: {sample_file.size} bytes")
        print(f"   Hash: {sample_file.content_hash[:16]}...")
        
        print(f"\n⏳ Before parsing:")
        print(f"   is_parsed():  {sample_file.is_parsed()}")
        print(f"   imports:      {sample_file.imports}")
        print(f"   functions:    {sample_file.functions}")
        print(f"   classes:      {sample_file.classes}")
        
        # Parse the file
        print(f"\n🔧 Parsing with tree-sitter...")
        success = parser.parse(sample_file)
        
        if success:
            print(f"✅ Parsing succeeded!")
            
            print(f"\n✨ After parsing:")
            print(f"   is_parsed():  {sample_file.is_parsed()}")
            print(f"   imports:      {len(sample_file.imports or [])} found")
            print(f"   functions:    {len(sample_file.functions or [])} found")
            print(f"   classes:      {len(sample_file.classes or [])} found")
            print(f"   docstrings:   {len(sample_file.docstrings or [])} found")
            
            # Show imports
            if sample_file.imports:
                print(f"\n📦 IMPORTS:")
                print("-" * 80)
                for i, imp in enumerate(sample_file.imports[:5], 1):
                    print(f"  {i}. {imp}")
                if len(sample_file.imports) > 5:
                    print(f"  ... and {len(sample_file.imports) - 5} more")
            
            # Show functions
            if sample_file.functions:
                print(f"\n🔧 FUNCTIONS:")
                print("-" * 80)
                for i, func in enumerate(sample_file.functions[:5], 1):
                    print(f"  {i}. {func['name']}{func.get('params', '()')}")
                    print(f"     Lines: {func['line_start']}-{func['line_end']}")
                    if func.get('decorators'):
                        print(f"     Decorators: {', '.join(func['decorators'])}")
                    if func.get('docstring'):
                        doc = func['docstring'][:60] + "..." if len(func['docstring']) > 60 else func['docstring']
                        print(f"     Doc: {doc}")
                    print()
                if len(sample_file.functions) > 5:
                    print(f"  ... and {len(sample_file.functions) - 5} more")
            
            # Show classes
            if sample_file.classes:
                print(f"\n📦 CLASSES:")
                print("-" * 80)
                for i, cls in enumerate(sample_file.classes[:3], 1):
                    print(f"  {i}. {cls['name']}")
                    if cls.get('bases'):
                        print(f"     Inherits: {', '.join(cls['bases'])}")
                    print(f"     Lines: {cls['line_start']}-{cls['line_end']}")
                    print(f"     Methods: {len(cls.get('methods', []))}")
                    if cls.get('docstring'):
                        doc = cls['docstring'][:60] + "..." if len(cls['docstring']) > 60 else cls['docstring']
                        print(f"     Doc: {doc}")
                    print()
                if len(sample_file.classes) > 3:
                    print(f"  ... and {len(sample_file.classes) - 3} more")
        else:
            print(f"❌ Parsing failed")
        
        # Test batch parsing
        print("\n" + "=" * 80)
        print("BATCH PARSING")
        print("=" * 80)
        
        # Parse first 10 Python files
        batch_files = python_files[:10]
        print(f"\n⏳ Parsing {len(batch_files)} Python files...")
        
        success_count = parser.parse_batch(batch_files)
        
        print(f"✅ Successfully parsed: {success_count}/{len(batch_files)}")
        
        # Show statistics
        print(f"\n📊 PARSING STATISTICS:")
        print("-" * 80)
        
        total_functions = sum(len(cf.functions or []) for cf in batch_files if cf.is_parsed())
        total_classes = sum(len(cf.classes or []) for cf in batch_files if cf.is_parsed())
        total_imports = sum(len(cf.imports or []) for cf in batch_files if cf.is_parsed())
        
        print(f"  Total functions:  {total_functions}")
        print(f"  Total classes:    {total_classes}")
        print(f"  Total imports:    {total_imports}")
        
        # Show the power of progressive enrichment
        print("\n" + "=" * 80)
        print("PROGRESSIVE ENRICHMENT DEMONSTRATION")
        print("=" * 80)
        
        if batch_files:
            cf = batch_files[0]
            print(f"\n📄 File: {cf.relative_path_str}")
            print(f"\n✅ FileScanner created CodeFile with:")
            print(f"   - absolute_path: {cf.absolute_path}")
            print(f"   - language: {cf.language.value}")
            print(f"   - size: {cf.size} bytes")
            print(f"   - content_hash: {cf.content_hash[:16]}...")
            print(f"   - repository_name: {cf.repository_name}")
            
            if cf.is_parsed():
                print(f"\n✅ Parser enriched SAME CodeFile with:")
                print(f"   - imports: {len(cf.imports or [])} modules")
                print(f"   - functions: {len(cf.functions or [])} functions")
                print(f"   - classes: {len(cf.classes or [])} classes")
                print(f"   - ast_tree: <tree-sitter AST>")
                
                print(f"\n⏳ Chunker will enrich SAME CodeFile with:")
                print(f"   - chunks: [list of semantic code units]")
                
                print(f"\n⏳ Embedder will enrich SAME CodeFile with:")
                print(f"   - embeddings: [vector embeddings for each chunk]")
                
                print(f"\n💡 Same object flows through entire pipeline!")
                print(f"   No data loss, easy debugging, efficient caching!")
    
    print("\n" + "=" * 80)
    print("✅ PARSER TEST COMPLETE")
    print("=" * 80)
    
    print("\n🎯 KEY DIFFERENTIATOR:")
    print("   ❌ Traditional RAG: Split text into 512-char chunks (arbitrary)")
    print("   ✅ Our approach: Parse AST, extract functions/classes (semantic)")
    print("\n   This is what makes the project interview-worthy! 🚀")


if __name__ == "__main__":
    test_parser()
