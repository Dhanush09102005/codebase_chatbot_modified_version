#!/usr/bin/env python3
"""
Test script demonstrating the improved ingestion pipeline.

Run this to see:
1. Repository cloning and caching
2. File scanning with Language enum
3. CodeFile objects with enriched metadata
4. File hashing for incremental indexing
5. Repository name in CodeFile
6. Private repo support
7. Refresh capability
"""

from pathlib import Path
from app.ingestion import RepositoryManager, FileScanner, CodeFile, Language


def test_ingestion():
    """Test the complete ingestion pipeline."""
    
    print("=" * 80)
    print("CODE Q&A ASSISTANT - INGESTION PIPELINE TEST")
    print("=" * 80)
    
    # Initialize components
    manager = RepositoryManager()
    scanner = FileScanner()
    
    # Test with FastAPI repository (small, well-organized)
    repo_url = "https://github.com/fastapi/fastapi.git"
    
    print(f"\n📥 Loading repository: {repo_url}")
    repo_path = manager.load(repo_url)
    print(f"✅ Repository stored at: {repo_path}")
    
    print(f"\n🔍 Scanning repository for code files...")
    code_files = scanner.scan(repo_path)
    
    print(f"✅ Found {len(code_files)} code files\n")
    
    # Show language breakdown
    print("📊 LANGUAGE BREAKDOWN (using Language enum):")
    print("-" * 80)
    language_counts = {}
    for cf in code_files:
        lang_str = cf.language.value  # Get enum value
        language_counts[lang_str] = language_counts.get(lang_str, 0) + 1
    
    for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang:20} {count:4} files")
    
    # Show sample files with NEW fields
    print("\n📝 SAMPLE CODE FILES (with hash and repo name):")
    print("-" * 80)
    for i, code_file in enumerate(code_files[:10]):
        size_kb = code_file.size / 1024
        hash_short = code_file.content_hash[:12]
        print(f"  {i+1:2}. {code_file.relative_path_str:40} | {code_file.language.value:12} | {size_kb:6.1f} KB | {hash_short}")
    
    # Show enhanced CodeFile structure
    if code_files:
        print("\n🔍 ENHANCED CODEFILE OBJECT (first file):")
        print("-" * 80)
        cf = code_files[0]
        print(f"  absolute_path:     {cf.absolute_path}")
        print(f"  relative_path:     {cf.relative_path}")
        print(f"  repository_root:   {cf.repository_root}")
        print(f"  repository_name:   {cf.repository_name}  ✨ NEW")
        print(f"  language:          {cf.language} (enum: {type(cf.language).__name__})  ✨ ENUM")
        print(f"  extension:         {cf.extension}")
        print(f"  size:              {cf.size} bytes")
        print(f"  content_hash:      {cf.content_hash[:16]}...  ✨ NEW (SHA-256)")
        print(f"  file_name:         {cf.file_name}")
        
        # Show parser readiness
        print(f"\n  Parser-ready fields:")
        print(f"    imports:         {cf.imports}  ✨ Ready for Parser")
        print(f"    functions:       {cf.functions}  ✨ Ready for Parser")
        print(f"    classes:         {cf.classes}  ✨ Ready for Parser")
        print(f"    docstrings:      {cf.docstrings}  ✨ Ready for Parser")
        print(f"    ast_tree:        {cf.ast_tree}  ✨ Ready for Parser")
        
        print(f"\n  Pipeline stages:")
        print(f"    is_parsed():     {cf.is_parsed()}")
        print(f"    is_chunked():    {cf.is_chunked()}")
        print(f"    is_embedded():   {cf.is_embedded()}")
    
    # Demo Language enum benefits
    print("\n🎯 LANGUAGE ENUM BENEFITS:")
    print("-" * 80)
    if code_files:
        cf = code_files[0]
        print(f"  Type-safe:        {cf.language} (type: {type(cf.language)})")
        print(f"  String value:     {cf.language.value}")
        print(f"  Autocomplete:     Language.PYTHON, Language.JAVASCRIPT, etc.")
        print(f"  No typos:         Language.PYTHON vs 'python' or 'Python' ✅")
        print(f"  Parser selection: if cf.language == Language.PYTHON: use_python_parser()")
    
    # Demo incremental indexing with hashes
    print("\n🔄 INCREMENTAL INDEXING (file hashing):")
    print("-" * 80)
    if code_files:
        cf = code_files[0]
        print(f"  File: {cf.relative_path_str}")
        print(f"  Hash: {cf.content_hash}")
        print(f"  Usage:")
        print(f"    1. Store hash in database")
        print(f"    2. On next scan, compare hashes")
        print(f"    3. If hash unchanged → skip parsing/embedding")
        print(f"    4. If hash changed → reprocess file")
        print(f"\n  Example check:")
        previous_hash = cf.content_hash
        print(f"    has_changed = cf.has_changed(previous_hash)  # → {cf.has_changed(previous_hash)}")
    
    # Get statistics
    print("\n📈 REPOSITORY STATISTICS:")
    print("-" * 80)
    stats = scanner.get_file_stats(repo_path)
    print(f"  Total files:       {stats['total_files']}")
    print(f"  Total size:        {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  Repository:        {stats['repository']}")
    print(f"  Repository name:   {stats['repository_name']}  ✨ NEW")
    
    # Show cached repositories with NEW metadata
    print("\n💾 CACHED REPOSITORIES (with metadata):")
    print("-" * 80)
    cached = manager.get_cached_repositories()
    if cached:
        for repo in cached:
            print(f"  {repo['name']}")
            print(f"    URL:          {repo['url']}")
            print(f"    Branch:       {repo['branch']}  ✨ NEW")
            print(f"    Clone time:   {repo['clone_time']}  ✨ NEW")
            print(f"    Last refresh: {repo['last_refresh']}  ✨ NEW")
            print(f"    Commit SHA:   {repo['commit_sha'][:12] if repo['commit_sha'] else 'N/A'}...  ✨ NEW")
    else:
        print("  No cached repositories")
    
    # Show metadata file example
    print("\n📄 METADATA MANIFEST (.repo_metadata.json):")
    print("-" * 80)
    print("  Each cached repository has a metadata file:")
    print("  {")
    print('    "url": "https://github.com/fastapi/fastapi.git",')
    print('    "branch": "main",')
    print('    "clone_time": "2024-01-15T10:30:00",')
    print('    "last_refresh_time": "2024-01-20T14:45:00",')
    print('    "commit_sha": "abc123...",')
    print('    "repository_name": "fastapi"')
    print("  }")
    print("  Benefits: Easy debugging, repo management, tracking")
    
    # Demo branch selection
    print("\n🌿 BRANCH SELECTION SUPPORT:")
    print("-" * 80)
    print("  Clone specific branch:")
    print("    manager.load(url, branch='develop')")
    print("    manager.load(url, branch='feature/new-api')")
    print("    manager.load(url, branch='v1.0.0')  # Tags work too")
    print("  Benefits:")
    print("    - Index feature branches")
    print("    - Compare different versions")
    print("    - Index release tags")
    
    # Demo secure authentication
    print("\n🔐 SECURE AUTHENTICATION (no token in URL):")
    print("-" * 80)
    print("  ✅ Uses Git credential helpers via environment variables")
    print("  ✅ Token NEVER appears in URL or logs")
    print("  ✅ Safe exception handling")
    print("  ❌ OLD (insecure): https://oauth:TOKEN@github.com/user/repo.git")
    print("  ✅ NEW (secure):   Uses GIT_USERNAME + GIT_PASSWORD env vars")
    
    # Demo safe refresh
    print("\n🛡️  SAFE REFRESH (cleanliness verification):")
    print("-" * 80)
    print("  Before pulling, checks for local modifications:")
    print("    if repo.is_dirty(untracked_files=True):")
    print("        logger.warning('Has local modifications, skipping')")
    print("        return False")
    print("  Benefits:")
    print("    - Prevents data loss")
    print("    - Safe for manual repo inspection")
    print("    - Can disable check: manager.refresh(path, verify_clean=False)")
    
    # Show repository info
    if repo_path:
        print("\n📊 REPOSITORY INFO:")
        print("-" * 80)
        info = manager.get_repository_info(repo_path)
        if info:
            print(f"  Path:         {info['path']}")
            print(f"  URL:          {info['url']}")
            print(f"  Branch:       {info['branch']}")
            print(f"  Commit SHA:   {info['commit_sha'][:12]}...")
            print(f"  Is dirty:     {info['is_dirty']} (has local mods)")
            print(f"  Clone time:   {info['clone_time']}")
            print(f"  Last refresh: {info['last_refresh']}")
    
    # Demo refresh capability
    print("\n🔄 REFRESH CAPABILITY:")
    print("-" * 80)
    print("  manager.load(url, refresh=True)            # Update cached repo")
    print("  manager.refresh(repo_path)                 # Pull latest changes")
    print("  manager.refresh(repo_path, verify_clean=False)  # Skip cleanliness check")
    
    # Demo private repo support
    print("\n🔐 PRIVATE REPOSITORY SUPPORT:")
    print("-" * 80)
    print("  manager = RepositoryManager(auth_token='ghp_xxxxx')")
    print("  manager.load('https://github.com/org/private.git')")
    print("  # Token used via environment variables, NOT embedded in URL")
    
    # Show why os.walk() is kept
    print("\n⚙️  WHY os.walk() IS KEPT:")
    print("-" * 80)
    print("  ✅ In-place directory filtering: dirs[:] = filtered_dirs")
    print("  ✅ More efficient than Path.rglob() for large repos")
    print("  ✅ Production pattern used in real scanners")
    print("  ✅ No recursion limits")
    print("  ❌ Path.rglob() can't prevent traversal into ignored dirs")
    
    print("\n" + "=" * 80)
    print("✅ INGESTION PIPELINE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_ingestion()
