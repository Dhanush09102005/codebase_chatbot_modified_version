# Code Q&A Assistant - Pipeline Architecture

## The Core Principle: Progressive Enrichment

**One data object flows through the entire pipeline, getting progressively enriched at each stage.**

This is the same pattern used in:
- **Compilers**: Source → Tokens → AST → IR → Machine Code
- **spaCy NLP**: Text → Doc → Tokens → POS → NER → Dependencies
- **TensorFlow**: Data → Preprocessing → Features → Model → Predictions

## The Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CODE Q&A ASSISTANT PIPELINE                   │
└─────────────────────────────────────────────────────────────────┘

1. RepositoryManager
   Input:  Git URL or local path
   Output: Path to repository
   ↓

2. FileScanner
   Input:  Repository path
   Output: List[CodeFile]
   
   CodeFile {
     ✅ absolute_path
     ✅ relative_path
     ✅ repository_name
     ✅ language (Language enum)
     ✅ extension
     ✅ size
     ✅ content_hash (SHA-256)
     ⏳ imports = None
     ⏳ functions = None
     ⏳ classes = None
     ⏳ docstrings = None
     ⏳ ast_tree = None
     ⏳ chunks = None
     ⏳ embeddings = None
   }
   ↓

3. Parser (tree-sitter)
   Input:  List[CodeFile]
   Output: List[CodeFile] (SAME OBJECTS, enriched)
   
   CodeFile {
     ✅ absolute_path
     ✅ relative_path
     ✅ repository_name
     ✅ language
     ✅ extension
     ✅ size
     ✅ content_hash
     ✅ imports = ["fastapi", "pydantic", ...]       ← NEW
     ✅ functions = [                                 ← NEW
          {"name": "get_user", "line_start": 10, "line_end": 25, ...},
          {"name": "create_user", "line_start": 27, "line_end": 45, ...}
        ]
     ✅ classes = [                                   ← NEW
          {"name": "User", "methods": [...], "line_start": 5, ...}
        ]
     ✅ docstrings = ["Get user by ID", ...]          ← NEW
     ✅ ast_tree = <tree-sitter AST>                  ← NEW
     ⏳ chunks = None
     ⏳ embeddings = None
   }
   ↓

4. Chunker (AST-aware)
   Input:  List[CodeFile]
   Output: List[CodeFile] (SAME OBJECTS, enriched)
   
   CodeFile {
     ✅ absolute_path
     ✅ relative_path
     ✅ repository_name
     ✅ language
     ✅ extension
     ✅ size
     ✅ content_hash
     ✅ imports
     ✅ functions
     ✅ classes
     ✅ docstrings
     ✅ ast_tree
     ✅ chunks = [                                    ← NEW
          {
            "text": "def get_user(id: int):\n    ...",
            "type": "function",
            "name": "get_user",
            "line_start": 10,
            "line_end": 25,
            "language": "python",
            "file_path": "app/main.py",
            "repo_name": "fastapi"
          },
          {
            "text": "class User:\n    ...",
            "type": "class",
            "name": "User",
            ...
          }
        ]
     ⏳ embeddings = None
   }
   ↓

5. Embedder (sentence-transformers)
   Input:  List[CodeFile]
   Output: List[CodeFile] (SAME OBJECTS, enriched)
   
   CodeFile {
     ✅ absolute_path
     ✅ relative_path
     ✅ repository_name
     ✅ language
     ✅ extension
     ✅ size
     ✅ content_hash
     ✅ imports
     ✅ functions
     ✅ classes
     ✅ docstrings
     ✅ ast_tree
     ✅ chunks
     ✅ embeddings = [                                ← NEW
          vector([0.123, -0.456, 0.789, ...]),  # 384-dim
          vector([0.234, -0.567, 0.890, ...]),
          ...
        ]
   }
   ↓

6. Indexer (ChromaDB)
   Input:  List[CodeFile]
   Output: Indexed in ChromaDB
   
   Each chunk stored with metadata:
   {
     "text": "def get_user(id: int): ...",
     "embedding": vector([...]),
     "metadata": {
       "file_path": "app/main.py",
       "repo_name": "fastapi",
       "language": "python",
       "type": "function",
       "name": "get_user",
       "line_start": 10,
       "line_end": 25,
       "content_hash": "abc123..."  # For incremental updates
     }
   }
   ↓

7. Retriever (Hybrid BM25 + Vector)
   Input:  User query
   Output: List[CodeFile] (with metadata.scores)
   
   CodeFile {
     ... (all fields from before)
     metadata = {                                     ← NEW
       "bm25_score": 0.85,
       "vector_similarity": 0.92,
       "combined_score": 0.89,
       "rank": 1,
       "query": "how to authenticate users"
     }
   }
   ↓

8. LLM Chain (LCEL + Llama 3.1)
   Input:  Query + List[CodeFile]
   Output: Answer with code references
```

## Why This Architecture?

### ✅ **No Data Loss**
Every stage can see everything from previous stages.

```python
# Parser can see file metadata
if code_file.size > 1_000_000:  # Large file
    logger.warning(f"Large file: {code_file.relative_path_str}")

# Chunker can see parser output
if len(code_file.functions) > 50:  # Many functions
    # Use different chunking strategy
    ...

# Embedder can see chunks + AST
for chunk in code_file.chunks:
    # Add docstring to chunk text for better embeddings
    if chunk["type"] == "function":
        docstring = find_docstring(code_file.docstrings, chunk["name"])
        chunk["text"] = f"{docstring}\n{chunk['text']}"
```

### ✅ **Easy Debugging**
Inspect what each stage added:

```python
# Check pipeline progress
print(f"Parsed: {code_file.is_parsed()}")       # True/False
print(f"Chunked: {code_file.is_chunked()}")     # True/False
print(f"Embedded: {code_file.is_embedded()}")   # True/False

# See what Parser found
print(f"Functions: {len(code_file.functions)}")
print(f"Classes: {len(code_file.classes)}")
print(f"Imports: {code_file.imports}")
```

### ✅ **Efficient Caching**
Cache the enriched object:

```python
# After parsing (expensive)
cache[code_file.content_hash] = code_file

# On next scan
new_file = scanner.scan(repo_path)[0]
if new_file.content_hash in cache:
    # File unchanged, reuse cached version
    code_file = cache[new_file.content_hash]
    # Skip parser, chunker (already done)
    embedder.embed(code_file)  # Only re-embed if needed
```

### ✅ **Incremental Indexing**
Only reprocess changed files:

```python
# Initial indexing
for code_file in code_files:
    parse(code_file)
    chunk(code_file)
    embed(code_file)
    db.store(code_file.content_hash, code_file)

# Later scan
new_files = scanner.scan(repo_path)
for new_file in new_files:
    old_hash = db.get_hash(new_file.relative_path_str)
    
    if new_file.has_changed(old_hash):
        # File changed, reprocess
        parse(new_file)
        chunk(new_file)
        embed(new_file)
        db.update(new_file)
    else:
        # File unchanged, skip
        logger.info(f"Skipping {new_file.file_name} (unchanged)")
```

### ✅ **Testable Stages**
Each stage is a pure transformation:

```python
# Test Parser
code_file = CodeFile(...)
parser.parse(code_file)
assert code_file.is_parsed()
assert len(code_file.functions) > 0

# Test Chunker
chunker.chunk(code_file)
assert code_file.is_chunked()
assert len(code_file.chunks) == len(code_file.functions) + len(code_file.classes)

# Test Embedder
embedder.embed(code_file)
assert code_file.is_embedded()
assert len(code_file.embeddings) == len(code_file.chunks)
```

## Key Differentiator: AST-Aware Chunking

### ❌ **Traditional RAG (what others do)**
```python
# Read file as text
text = open("main.py").read()

# Split by character count (arbitrary)
chunks = [
    text[0:512],      # Might split mid-function
    text[512:1024],   # Might split mid-class
    text[1024:1536],  # No semantic meaning
    ...
]

# Embed and index
embeddings = embed(chunks)
```

**Problems:**
- Splits mid-function/class (loses context)
- No understanding of code structure
- Can't filter by "show me all functions"
- No line numbers for linking back to code

### ✅ **Our Approach (AST-aware)**
```python
# Parse with tree-sitter
parser.parse(code_file)  # Extracts functions, classes, imports

# Chunk by semantic units
chunks = [
    {
        "text": "def get_user(id: int):\n    ...",
        "type": "function",
        "name": "get_user",
        "line_start": 10,
        "line_end": 25,
    },
    {
        "text": "class User:\n    def __init__(self): ...",
        "type": "class",
        "name": "User",
        "line_start": 27,
        "line_end": 50,
    }
]

# Each chunk is a complete semantic unit
```

**Benefits:**
- ✅ Each chunk is a complete function/class
- ✅ Preserves code structure
- ✅ Can filter: "show me functions named 'get_*'"
- ✅ Line numbers for jump-to-definition
- ✅ Better embeddings (semantic units, not arbitrary splits)

## Interview Talking Points

1. **"I used a progressive enrichment pipeline"**
   → Same pattern as spaCy, TensorFlow, production compilers

2. **"AST-aware chunking, not arbitrary text splitting"**
   → This is the key differentiator from other Code RAG projects

3. **"Incremental indexing with content hashing"**
   → Only reprocess changed files, efficient for large codebases

4. **"Single data model through entire pipeline"**
   → Easy debugging, no data loss, testable stages

5. **"tree-sitter for parsing"**
   → Industry standard, supports 40+ languages, used by GitHub, Neovim

---

**Next: Build the Parser module with tree-sitter** 🚀
