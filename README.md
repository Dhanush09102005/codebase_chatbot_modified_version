# Code Q&A Assistant

RAG-based chatbot for answering questions about codebases using semantic search and LLM generation.

## Features

- **AST-Based Parsing**: Semantic code understanding (not arbitrary text splitting)
- **Semantic Chunking**: Functions, classes, methods as chunks (preserves context)
- **Hybrid Retrieval**: Vector similarity + BM25 keyword search + reranking
- **Multi-Provider LLM**: HuggingFace, Groq, OpenAI, Ollama
- **Progressive Enrichment**: Single object flows through entire pipeline
- **Production-Ready**: FastAPI, proper error handling, logging, metrics

## Architecture

```
INDEXING (Offline):
  Repository → Scanner → Parser → Chunker → Embedder → Indexer → ChromaDB

RAG CHAIN (Online):
  Question → Retriever → PromptBuilder → LLMClient → Answer

API LAYER:
  HTTP Request → FastAPI → Services → HTTP Response
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PransuVadsmiya/demo_codebase_chatbot.git
   cd demo_codebase_chatbot
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Configuration

Edit `.env` file:

```env
# LLM Provider
LLM_PROVIDER=huggingface  # or groq, openai, ollama
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct

# API Keys (as needed)
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Retrieval Settings
RETRIEVAL_TOP_K=10
RERANK_TOP_K=5
```

## Usage

### Start the API Server

```bash
python main.py
# Or
uvicorn app.api.app:app --reload
```

Server runs at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### API Endpoints

#### 1. Index a Repository
```bash
curl -X POST http://localhost:8000/api/v1/repositories/index \
  -H "Content-Type: application/json" \
  -d '{
    "repository_url": "https://github.com/fastapi/fastapi.git"
  }'
```

#### 2. Ask Questions
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does FastAPI handle dependencies?"
  }'
```

#### 3. List Repositories
```bash
curl http://localhost:8000/api/v1/repositories
```

#### 4. Delete Repository
```bash
curl -X DELETE http://localhost:8000/api/v1/repositories/{repository_id}
```

#### 5. Health Check
```bash
curl http://localhost:8000/api/v1/health
```

## Project Structure

```
.
├── app/
│   ├── core/                 # Configuration, logging
│   ├── ingestion/           # Repository loading, file scanning
│   ├── rag/                 # RAG pipeline components
│   │   ├── parser.py        # AST-based code parsing
│   │   ├── ast_chunker.py   # Semantic chunking
│   │   ├── embedder.py      # Embedding pipeline
│   │   ├── indexer.py       # Vector store persistence
│   │   ├── retriever.py     # Hybrid retrieval
│   │   ├── llm_client.py    # Multi-provider LLM clients
│   │   ├── prompt_builder.py # Prompt formatting
│   │   └── rag_chain.py     # RAG orchestration
│   └── api/                 # FastAPI endpoints
├── main.py                  # Application entry point
├── requirements.txt         # Dependencies
└── .env.example            # Configuration template
```

## Key Design Principles

### 1. Single Responsibility
- **Retriever**: Only retrieves chunks (no prompting, no LLM)
- **PromptBuilder**: Only formats prompts (no retrieval, no LLM)
- **LLMClient**: Only calls LLM APIs (no retrieval, no prompting)
- **RAGChain**: Only orchestrates (no implementation)

### 2. Progressive Enrichment
Same `CodeFile` object flows through entire pipeline:
```
CodeFile (basic) 
  → + AST (Parser)
  → + chunks (Chunker)
  → + embeddings (Embedder)
  → persisted (Indexer)
```

### 3. Interface Abstraction
- Swap embedding models: `sentence-transformers` ↔ `HuggingFace`
- Swap vector stores: `ChromaDB` ↔ `Milvus` ↔ `Qdrant`
- Swap LLM providers: `HuggingFace` ↔ `Groq` ↔ `OpenAI` ↔ `Ollama`

## Testing

Run test scripts to verify components:

```bash
# Test indexing pipeline
python test_indexer.py

# Test RAG chain
python test_rag_chain.py

# Test API overview
python test_api.py
```

## Tech Stack

- **Framework**: FastAPI, Pydantic
- **LLM**: HuggingFace Transformers, Groq, OpenAI, Ollama
- **Embeddings**: sentence-transformers
- **Vector Store**: ChromaDB
- **Parsing**: tree-sitter
- **Retrieval**: Hybrid (vector + BM25 + reranking)
- **Repository Management**: GitPython

## Performance

- **Incremental Indexing**: Skip unchanged files
- **Batch Processing**: Efficient embedding generation
- **Shallow Cloning**: Faster repository loading
- **Metadata Filtering**: Query optimization

## Contributing

Contributions welcome! This project demonstrates:
- Clean architecture (SOLID principles)
- Production patterns (error handling, logging, metrics)
- Type safety (Pydantic models throughout)
- Extensibility (easy to add new providers)

## License

MIT License - See LICENSE file for details

## Author

Built by Pransu Vadsmiya

## Acknowledgments

- Built on FastAPI, ChromaDB, sentence-transformers
- Inspired by production RAG systems
- Designed for interview demonstrations and portfolio
