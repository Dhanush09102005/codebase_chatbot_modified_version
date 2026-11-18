# Code Q&A Assistant — System Design

## 1. Purpose and Scope

The Code Q&A Assistant is a retrieval-augmented generation (RAG) system for answering questions about one or more Git repositories. It:

- clones or caches Git repositories;
- scans supported source files;
- parses code into AST-aware semantic units;
- embeds and persists code chunks in ChromaDB;
- performs metadata-filtered, hybrid vector and keyword retrieval;
- constructs a grounded prompt with conversation memory;
- generates a cited answer with a configurable HuggingFace, Groq, OpenAI, or Ollama client.

The implementation is a two-service Docker deployment: a React/Nginx frontend and a FastAPI backend. ChromaDB and cached repositories are local persistent volumes.

## 2. System Context

```mermaid
flowchart LR
    User[Developer or technical user]
    UI[React web application]
    API[FastAPI application]
    Git[GitHub, GitLab, or Git server]
    Chroma[(ChromaDB<br/>code_chunks)]
    Embeddings[Sentence-transformer model]
    LLM[LLM provider]
    Disk[(Local storage<br/>data/repos<br/>data/chroma_db)]

    User -->|Ask questions / manage repositories| UI
    UI -->|HTTP JSON over /api/v1| API
    API -->|Clone, pull, or read| Git
    API -->|Parse and scan| RepoCache[Repository cache]
    API -->|Embed chunks and queries| Embeddings
    API -->|Index and retrieve| Chroma
    API -->|Generate answers| LLM
    RepoCache --> Disk
    Chroma --> Disk
```

## 3. Container Architecture

```mermaid
flowchart TB
    subgraph DockerHost["Docker host"]
        subgraph FrontendContainer["web container"]
            Nginx[Nginx<br/>static React build]
        end

        subgraph BackendContainer["api container"]
            FastAPI[FastAPI application]
            Routes[API routes]
            Ingestion[Repository and file ingestion]
            RAG[RAG pipeline]
            Core[Settings and logging]
        end

        subgraph Persistent["Mounted volumes"]
            RepoData[(data/repos)]
            VectorData[(data/chroma_db)]
            Logs[(logs)]
        end
    end

    Browser[Browser] -->|Port 3000| Nginx
    Nginx -->|/api proxy| FastAPI
    FastAPI --> Routes
    Routes --> Ingestion
    Routes --> RAG
    Routes --> Core
    Ingestion --> RepoData
    RAG --> VectorData
    Core --> Logs
```

The frontend container does not perform business logic. Nginx serves `frontend/dist` and proxies `/api` to the `api:8000` Docker service.

## 4. High-Level Component Diagram

```mermaid
flowchart LR
    subgraph API["app/api"]
        App[app.py]
        Routes[routes.py]
        Models[Pydantic models]
    end

    subgraph Core["app/core"]
        Config[Settings]
        Logger[Rotating logging]
    end

    subgraph Ingestion["app/ingestion"]
        Repo[RepositoryManager]
        Metadata[RepositoryMetadata]
        Scanner[FileScanner]
        Language[Language detection]
        CodeFile[CodeFile model]
    end

    subgraph RAG["app/rag"]
        Parser[CodeParser / tree-sitter]
        Chunker[ASTChunker]
        Embedder[CodeEmbedder]
        Indexer[CodeIndexer]
        Retriever[CodeRetriever]
        Prompt[PromptBuilder]
        LLM[LLMClient implementations]
        Chain[RAGChain]
        Store[VectorStore interface]
        Chroma[ChromaVectorStore]
    end

    App --> Routes
    Routes --> Models
    Routes --> Config
    Routes --> Logger
    Routes --> Repo
    Routes --> Scanner
    Routes --> Parser
    Routes --> Chunker
    Routes --> Embedder
    Routes --> Indexer
    Routes --> Retriever
    Routes --> Prompt
    Routes --> LLM
    Routes --> Chain
    Repo --> Metadata
    Scanner --> Language
    Scanner --> CodeFile
    Parser --> CodeFile
    Chunker --> CodeFile
    Embedder --> CodeFile
    Indexer --> Store
    Retriever --> Store
    Retriever --> Embedder
    Store --> Chroma
    Chain --> Retriever
    Chain --> Prompt
    Chain --> LLM
```

## 5. Indexing Pipeline

```mermaid
sequenceDiagram
    actor User
    participant API as FastAPI route
    participant RM as RepositoryManager
    participant FS as FileScanner
    participant CP as CodeParser
    participant AC as ASTChunker
    participant CE as CodeEmbedder
    participant CI as CodeIndexer
    participant Git as Git remote
    participant Disk as data/repos
    participant Chroma as ChromaDB

    User->>API: POST /repositories/index
    API->>RM: load(url, branch, refresh)
    alt Repository is uncached
        RM->>Git: shallow clone
        Git-->>RM: files and commit SHA
        RM->>Disk: save clone and .repo_metadata.json
    else Cached and refresh=false
        RM->>Disk: reuse repository
    else Cached and refresh=true
        RM->>Git: pull or reclone
        RM->>Disk: update cache
    end

    RM-->>API: repository path
    API->>FS: scan(path, repo name/id)
    FS->>Disk: walk files, filter directories, hash SHA-256
    FS-->>API: CodeFile objects

    API->>CP: parse_batch(CodeFiles)
    CP->>Disk: read source text
    CP->>CP: tree-sitter AST extraction
    CP-->>API: parsed CodeFiles

    API->>AC: chunk_batch(CodeFiles)
    AC->>AC: create import/function/class chunks
    AC-->>API: chunks attached to CodeFiles

    API->>CE: embed_files(CodeFiles)
    CE->>CE: batch encode chunk text plus metadata
    CE-->>API: chunks with normalized vectors

    API->>CI: index_files(CodeFiles)
    CI->>CI: stable chunk IDs, metadata preparation
    CI->>Chroma: batch upsert
    Chroma-->>CI: indexed/skipped/failed counts
    CI-->>API: indexing statistics
    API-->>User: repository ID, files, chunks, timing
```

### Progressive enrichment

The central indexing data model is `CodeFile`. The same object is progressively enriched:

```text
CodeFile
  + absolute/relative path
  + repository_id/name
  + language and SHA-256 content hash
  -> AST and extracted imports/functions/classes/docstrings
  -> semantic CodeChunk objects
  -> embedding vectors
  -> persisted Chroma documents
```

Stable chunk IDs use repository ID, file path, chunk type, name, and start line. Chunks are skipped when that ID already exists.

## 6. Query Pipeline

```mermaid
sequenceDiagram
    actor User
    participant UI as React chat page
    participant API as FastAPI /chat
    participant RC as RAGChain
    participant CR as CodeRetriever
    participant EM as Embedding model
    participant Chroma as ChromaDB
    participant PB as PromptBuilder
    participant LLM as Configured LLM

    User->>UI: submit question and optional filters
    UI->>API: POST /chat + chat history
    API->>RC: query(question, filters, history)
    RC->>RC: expand retrieval query with recent code terms

    RC->>CR: retrieve(expanded query, filters)
    CR->>EM: embed query
    EM-->>CR: normalized query vector

    par Vector search
        CR->>Chroma: cosine query + metadata where
        Chroma-->>CR: top vector candidates
    and Keyword search
        CR->>Chroma: read filtered documents
        CR->>CR: tokenize and BM25-style scoring
    end

    CR->>CR: merge by chunk ID
    CR->>CR: code-aware reranking and related chunk expansion
    CR-->>RC: top retrieved chunks and stats

    alt No relevant chunks
        RC-->>API: grounded no-context response
    else Chunks found
        RC->>PB: build prompt(question, chunks, history)
        PB-->>RC: strict grounded prompt
        RC->>LLM: generate(prompt, tokens, temperature)
        LLM-->>RC: answer and usage metadata
    end

    RC-->>API: RAGResponse
    API-->>UI: answer, sources, timing, usage
    UI-->>User: rendered answer and source references
```

The frontend sends the previous six turns as conversation memory. `RAGChain` uses them to make an expanded deterministic retrieval query and `PromptBuilder` includes them only as conversational context, not as code evidence.

## 7. Retrieval and Reranking Design

```mermaid
flowchart TD
    Q[User question] --> EQ[Expand follow-up terms]
    EQ --> VE[Vector embedding]
    EQ --> KT[Keyword tokenization]
    VE --> VS[Chroma vector search]
    KT --> BM25[BM25-style candidate scoring]
    VS --> Merge[Merge by chunk ID]
    BM25 --> Merge
    Merge --> Scores[Hybrid alpha combination]
    Scores --> Rerank[Code-aware reranking]
    Rerank --> Related[Expand related class/method context]
    Related --> Rerank2[Second reranking pass]
    Rerank2 --> Threshold[Minimum-score filter]
    Threshold --> TopK[Top rerank_k chunks]
    TopK --> Prompt[Prompt construction]
```

Retrieval uses repository, language, file, and chunk metadata. Code-aware reranking adds identifier overlap, exact identifier matches, chunk-type boosts, constructor/method intent, source-file preference over tests, and docstring boosts.

## 8. Deployment View

```mermaid
flowchart LR
    subgraph UserMachine["Developer machine / Docker host"]
        Browser[Browser<br/>localhost:3000]

        subgraph Compose["docker compose"]
            Nginx[web<br/>Nginx + React<br/>port 3000:80]
            API[api<br/>FastAPI<br/>port 8000]
        end

        RepoVolume[./data/repos]
        ChromaVolume[./data/chroma_db]
        LogVolume[./logs]
        Env[.env]

        Browser --> Nginx
        Nginx -->|/api -> api:8000| API
        API --> RepoVolume
        API --> ChromaVolume
        API --> LogVolume
        Env -. environment .-> API
    end

    Git[Git remote] -->|clone/pull| API
    LLM[LLM provider API or Ollama] -->|generation| API
    HuggingFace[HuggingFace model storage] -->|local model download/load| API
```

Ports:

| Component | Host port | Container port |
|---|---:|---:|
| Frontend / Nginx | 3000 | 80 |
| Backend / FastAPI | 8000 | 8000 |

Docker volumes map `./data` and `./logs` into the API container. The persistent vector database and cloned repositories survive container restarts.

## 9. API Surface

| Method | Path | Responsibility |
|---|---|---|
| `GET /` | API metadata | Basic service description |
| `GET /health` | `/health` | Service liveness |
| `GET /api/v1/health` | Health detail | Chroma, embedding model, and configured LLM status |
| `POST /api/v1/repositories/index` | Repository indexing | Clone/cache, scan, parse, chunk, embed, and index |
| `GET /api/v1/repositories` | Repository inventory | Read cached repository metadata |
| `DELETE /api/v1/repositories/{repository_id}` | Repository deletion | Remove vectors and cached repository |
| `POST /api/v1/chat` | Code Q&A | Hybrid retrieval, prompt, and generation |

FastAPI also exposes interactive API documentation at `/docs` and `/redoc`.

## 10. Data Model

```mermaid
erDiagram
    REPOSITORY_METADATA {
        string repository_id PK
        string repository_name
        string url
        string branch
        string commit_sha
        string clone_time
        string last_refresh_time
    }

    CODE_FILE {
        string absolute_path
        string relative_path
        string repository_id FK
        string repository_name
        string language
        string content_hash
        int size
    }

    CODE_CHUNK {
        string chunk_id PK
        string repository_id FK
        string repository_name
        string file_path
        string language
        string chunk_type
        string chunk_name
        int line_start
        int line_end
        string text
        float vector
    }

    RETRIEVAL_RESULT {
        string chunk_id FK
        float vector_score
        float bm25_score
        float combined_score
        float rerank_score
    }

    REPOSITORY_METADATA ||--o{ CODE_FILE : contains
    CODE_FILE ||--o{ CODE_CHUNK : produces
    CODE_CHUNK ||--o{ RETRIEVAL_RESULT : participates_in
```

Repository metadata is stored as `.repo_metadata.json` under each cached repository directory. ChromaDB stores chunk text, vectors, and filterable metadata.

## 11. Configuration

Important configuration groups:

- **LLM:** provider, model, token, device, and Ollama URL.
- **Embeddings:** model, batch size, dimensions, normalization, and device.
- **Retrieval:** initial candidate count, rerank count, hybrid alpha, minimum score, and reranking toggle.
- **Prompt:** included line numbers/types, maximum chunks, and instruction style.
- **ChromaDB:** persistent directory, collection name, and distance metric.
- **Repository storage:** cache directory, GitHub token, and hash algorithm.
- **Scanning:** supported extensions and ignored directories/files.
- **Operations:** API host/port, log directory, log size, and backup count.

Secrets must remain in `.env` or deployment environment variables and must not be embedded in repository URLs or source logs.

## 12. Reliability and Failure Behavior

- Invalid or inaccessible repository sources return an indexing HTTP 500 through the route's error handling.
- No code files found returns HTTP 400.
- Repository deletion validates that the target path remains inside the configured cache directory.
- Pull failures attempt a clean repository refresh/reclone path.
- Vector, embedding, and LLM health are reported separately.
- A query with no retrieved chunks returns a no-context answer without calling the LLM.
- Component exceptions are logged and surfaced at API boundaries.

## 13. Security and Operational Boundaries

The current implementation has no authentication, authorization, per-user isolation, or rate limiting. CORS currently allows every origin. For production use, the design should add:

- authentication and authorization;
- strict CORS allowlists;
- repository URL validation and SSRF controls;
- request size and timeout limits;
- asynchronous or queued indexing for large repositories;
- encrypted or managed secret storage;
- TLS termination;
- audit logs and tenant-scoped collections or metadata.

Repository cloning uses Git credential helpers or an environment-provided token rather than placing the token in the Git URL.

## 14. Scalability and Evolution

The implementation is suitable for a local or small-team deployment but is not horizontally scalable in its current form:

- ChromaDB is on local persistent storage.
- Repository caches and metadata are local filesystem state.
- FastAPI creates pipeline components per route request.
- HuggingFace models may be loaded into process memory repeatedly.
- Indexing and querying are synchronous HTTP operations.
- Chat history is client-provided; no server-side session database exists.

Recommended evolution:

1. add a job queue and worker pool for indexing;
2. persist repository records and index jobs in PostgreSQL;
3. use a managed or distributed vector database;
4. cache embedding models and provider clients in long-lived workers;
5. use streaming responses for chat;
6. add incremental file diffing instead of stable-ID skipping alone;
7. add authentication and tenant-aware retrieval filters;
8. add parser implementations for JavaScript, Java, Go, Rust, and other declared languages.

## 15. Current Implementation Notes

These points distinguish the implemented system from aspirational features:

- JavaScript, Java, Go, and Rust parser branches are currently placeholders; unsupported extraction falls back to text chunking.
- The retriever has structure-aware reranking and related-context expansion.
- Repository index metadata tracks files and chunks, but full-file and chunk status is not stored in a separate database.
- Incremental indexing primarily skips existing stable chunk IDs; deleted or changed old chunks are not globally reconciled during re-indexing.
- LLM, embedding, and Ollama integrations are provider abstractions, but no provider is abstracted for the vector store interface beyond ChromaDB.
- The UI supports repository selection, language filtering, source citations, chat memory, and health/repository pages.
