# Code Q&A Assistant

RAG-based chatbot for answering questions about codebases using semantic search and LLM generation.

## Features

- **AST-Based Parsing**: Semantic code understanding (not arbitrary text splitting)
- **Semantic Chunking**: Functions, classes, methods as chunks (preserves context)
- **Hybrid Retrieval**: Vector similarity + BM25 keyword search + reranking
- **Multi-Provider LLM**: HuggingFace, Groq, OpenAI, Ollama
- **Progressive Enrichment**: Single object flows through entire pipeline
- **Chat Session Memory**: Persistent conversation history with up to 10 saved sessions
- **Global Exception Handling**: Structured error responses with traceable error IDs
- **Request/Response Logging**: Per-request tracing with duration and status
- **Rate Limiting**: Per-IP sliding window limits to prevent abuse
- **Input Sanitization**: Validated and sanitized inputs on all endpoints

## Architecture

```
INDEXING (Offline):
  Repository → Scanner → Parser → Chunker → Embedder → Indexer → ChromaDB

RAG CHAIN (Online):
  Question → Retriever → PromptBuilder → LLMClient → Answer

CHAT SESSIONS:
  POST /sessions → session_id
  POST /chat (with session_id) → answer saved to SQLite history
  GET /sessions/{id} → full conversation replay

API LAYER:
  HTTP Request → Rate Limiter → Sanitization → FastAPI → Services → HTTP Response
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

#### 6. Create a Chat Session
```bash
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "FastAPI deep dive"}'
```

Returns a `session_id`. Pass this in subsequent `/chat` requests to maintain conversation context.

#### 7. Chat with Session Memory
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does dependency injection work?",
    "session_id": "<session_id from above>"
  }'
```

#### 8. List Saved Sessions
```bash
curl http://localhost:8000/api/v1/sessions
```

#### 9. Get Full Session History
```bash
curl http://localhost:8000/api/v1/sessions/{session_id}
```

#### 10. Delete a Session
```bash
curl -X DELETE http://localhost:8000/api/v1/sessions/{session_id}
```

## Security & Reliability

### 1. Global Exception Handler
All unhandled exceptions are caught and returned as structured JSON:
```json
{
  "error": "internal_server_error",
  "message": "An unexpected error occurred. Please try again later.",
  "error_id": "a3f5e8d1"
}
```
Full traceback is logged to `app.log` with the same `error_id` for easy correlation. Special handling for timeout errors returns 504 Gateway Timeout.

### 2. Request/Response Logging
Every request and response is logged with:
- Unique request ID (also returned in `X-Request-ID` header)
- Client IP address
- HTTP method and path
- Status code and duration (in milliseconds)

Example log:
```
[a3f5e8d1] → POST /api/v1/chat from 127.0.0.1
[a3f5e8d1] ← 200 POST /api/v1/chat (1245.3ms)
```

### 3. Rate Limiting
Per-IP sliding window rate limits (60-second window):
- **Indexing** (`/repositories/index`): 3 requests/minute
- **Chat** (`/chat`): 20 requests/minute
- **Everything else**: 60 requests/minute

When exceeded, returns `429 Too Many Requests` with `Retry-After: 60` header.

### 4. Input Sanitization
All user inputs are validated using Pydantic validators:

**Repository URLs:**
- Must be valid Git URLs (HTTPS/SSH)
- Blocks `file://` and path traversal attempts

**Branch names:**
- Alphanumeric + safe characters only
- Blocks shell injection patterns (`;`, `|`, `` ` ``)

**Questions:**
- Length: 3-2000 characters
- Excessive repetition detection (DoS prevention)
- Whitespace normalization

**Session IDs:**
- Must be valid UUID format

**Repository/file names:**
- Blocks path traversal (`../`, `..\\`)
- Alphanumeric validation

Invalid inputs return `422 Unprocessable Entity` before reaching endpoint handlers.

## Project Structure

```
.
├── app/
│   ├── core/                 # Configuration, logging
│   ├── ingestion/           # Repository loading, file scanning
│   ├── chat/                # Chat session storage (SQLite)
│   │   ├── chat_store.py    # Session CRUD operations
│   │   └── __init__.py
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
│       ├── app.py           # Application + middleware
│       ├── routes.py        # Endpoint handlers
│       └── models.py        # Request/response models
├── data/
│   ├── chroma_db/           # ChromaDB vector storage
│   ├── repos/               # Cloned repository cache
│   └── chat_sessions.db     # SQLite session storage
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

## Chat Session Memory

The system supports **persistent conversation history** with up to **10 saved sessions**. Sessions are stored in SQLite and automatically include previous messages as context.

### How It Works

1. **Create a session**: `POST /sessions` returns a `session_id`
2. **Chat with memory**: Pass the `session_id` in `/chat` requests
3. **Context injection**: Last 10 messages are automatically injected into the LLM prompt
4. **Auto-titling**: First user message becomes the session title
5. **Auto-eviction**: When the 11th session is created, the oldest is deleted

### Example Workflow

```bash
# 1. Create session
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{}' | jq -r '.session_id')

# 2. Ask follow-up questions with context
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"question\": \"How does FastAPI handle dependencies?\",
    \"session_id\": \"$SESSION_ID\"
  }"

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"question\": \"Can you show me an example?\",
    \"session_id\": \"$SESSION_ID\"
  }"

# 3. Review history
curl http://localhost:8000/api/v1/sessions/$SESSION_ID
```

The second question ("Can you show me an example?") is answered with full awareness of the first question about dependency injection.

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
