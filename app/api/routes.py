"""
API routes for the Code Q&A system.

Thin interface over existing services - no business logic here.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import time
import json
from app.core import logger, settings
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
from app.api.models import (
    IndexRepositoryRequest,
    IndexRepositoryResponse,
    ChatRequest,
    ChatResponse,
    SourceReference,
    RepositoryInfo,
    ListRepositoriesResponse,
    DeleteRepositoryResponse,
    HealthCheck,
)


router = APIRouter()


# ==============================================================================
# Repository Indexing
# ==============================================================================

@router.post(
    "/repositories/index",
    response_model=IndexRepositoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Index a repository",
    description="Clone (or load cached) repository, scan files, parse, chunk, embed, and index."
)
async def index_repository(request: IndexRepositoryRequest):
    """
    Index a Git repository for code Q&A.
    
    This endpoint orchestrates the complete indexing pipeline:
    1. Clone/load repository
    2. Scan for code files
    3. Parse AST
    4. Chunk semantically
    5. Generate embeddings
    6. Index in vector store
    """
    logger.info(f"Indexing request: {request.repository_url}")
    start_time = time.time()
    
    try:
        # Initialize pipeline components
        manager = RepositoryManager()
        scanner = FileScanner()
        parser = CodeParser()
        chunker = ASTChunker()
        embedder = CodeEmbedder()
        indexer = CodeIndexer()
        
        # Step 1: Load repository
        logger.info("Step 1/6: Loading repository...")
        repo_path = manager.load(
            source=request.repository_url,
            refresh=request.force_refresh,
            branch=request.branch
        )
        
        # Get repository info and ensure it has required fields
        repo_info = manager.get_repository_info(repo_path)
        logger.info(f"repo_info from get_repository_info: {repo_info}")
        
        # Extract required fields regardless of whether we got info from metadata
        repo_id = manager._hash_url(request.repository_url)
        logger.info(f"repo_id computed: {repo_id}")
        repo_name = manager._extract_repo_name(request.repository_url)
        logger.info(f"repo_name computed: {repo_name}")
        
        if not repo_info:
            repo_info = {}
        
        # Ensure repo_info has all required fields
        repo_info['repo_id'] = repo_id
        repo_info['repo_name'] = repo_name
        repo_info['repo_url'] = request.repository_url
        if 'branch' not in repo_info:
            repo_info['branch'] = request.branch or 'main'
        logger.info(f"repo_info after enrichment: {repo_info}")
        
        # Step 2: Scan files
        logger.info("Step 2/6: Scanning files...")
        code_files = scanner.scan(repo_path,repository_name=repo_info["repo_name"])
            # code_files = scanner.scan(repo_path)
        
        if not code_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No code files found in repository"
            )
        
        logger.info(f"Found {len(code_files)} code files")
        
        # Step 3: Parse
        logger.info("Step 3/6: Parsing AST...")
        parsed_count = parser.parse_batch(code_files)
        logger.info(f"Parsed {parsed_count} files")
        
        # Step 4: Chunk
        logger.info("Step 4/6: Chunking...")
        chunked_count = chunker.chunk_batch(code_files)
        logger.info(f"Chunked {chunked_count} files")
        
        # Step 5: Embed
        logger.info("Step 5/6: Embedding...")
        embedded_count = embedder.embed_files(code_files)
        logger.info(f"Embedded {embedded_count} files")
        
        # Step 6: Index
        logger.info("Step 6/6: Indexing...")
        index_stats = indexer.index_files(code_files)
        
        elapsed = time.time() - start_time
        
        logger.info(
            f"Indexing complete: {index_stats.indexed_chunks} chunks "
            f"in {elapsed:.2f}s"
        )
        
        return IndexRepositoryResponse(
            repository_id=repo_info['repo_id'],
            repository_name=repo_info['repo_name'],
            status="success",
            files_processed=len(code_files),
            chunks_indexed=index_stats.indexed_chunks,
            indexing_time_seconds=elapsed,
            metadata={
                "repo_path": str(repo_path),
                "branch": repo_info.get('branch', 'main'),
                "parsed_files": parsed_count,
                "chunked_files": chunked_count,
                "embedded_files": embedded_count,
                "skipped_chunks": index_stats.skipped_chunks,
                "failed_chunks": index_stats.failed_chunks,
            }
        )
    
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )


# ==============================================================================
# Chat / Q&A
# ==============================================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question about code",
    description="Run RAG chain to answer question using indexed code."
)
async def chat(request: ChatRequest):
    """
    Answer a question about indexed code.
    
    This endpoint:
    1. Retrieves relevant code chunks
    2. Builds structured prompt
    3. Generates answer with LLM
    4. Returns answer with sources
    """
    logger.info(f"Chat request: {request.question[:100]}...")
    
    try:
        # Initialize RAG chain
        retriever = CodeRetriever()
        prompt_builder = PromptBuilder()
        
        # Create LLM client based on settings
        llm_client = create_llm_client(
            provider=settings.llm_provider,
            model_name=settings.llm_model,
            api_key=getattr(settings, f"{settings.llm_provider}_api_key", None),
        )
        
        rag_chain = RAGChain(
            retriever=retriever,
            prompt_builder=prompt_builder,
            llm_client=llm_client
        )
        
        # Build filters
        filters = None
        if request.repository_name or request.language:
            filters = RetrievalFilters(
                repository_name=request.repository_name,
                language=request.language,
            )
        
        # Query
        response = rag_chain.query(
            question=request.question,
            filters=filters,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Convert to API response
        sources = [
            SourceReference(
                repository_name=result.repository_name or "unknown",
                file_path=result.file_path,
                chunk_type=result.chunk_type,
                chunk_name=result.chunk_name,
                line_start=result.line_start,
                line_end=result.line_end,
                relevance_score=result.combined_score
            )
            for result in response.retrieval_context.results
        ]
        
        # print("=" * 60)
        # print("Provider:", settings.llm_provider)
        # print("Model:", settings.llm_model)
        # print("Token:", "YES" if settings.hf_token else "NO")
        # print("=" * 60)
        return ChatResponse(
            question=response.question,
            answer=response.answer,
            sources=sources,
            retrieval_stats={
                "chunks_retrieved": len(response.retrieval_context.results),
                "retrieval_time_seconds": response.retrieval_context.retrieval_time_seconds,
            },
            llm_stats={
                "model": response.llm_response.model,
                "tokens_used": response.llm_response.tokens_used,
                "prompt_tokens": response.llm_response.prompt_tokens,
                "completion_tokens": response.llm_response.completion_tokens,
                "llm_time_seconds": response.llm_response.time_seconds,
            },
            total_time_seconds=response.total_time_seconds
        )
    
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )


# ==============================================================================
# Repository Management
# ==============================================================================

@router.get(
    "/repositories",
    response_model=ListRepositoriesResponse,
    summary="List indexed repositories",
    description="Get all indexed repositories with statistics."
)
async def list_repositories():
    """
    List all indexed repositories.
    
    Returns repository information and indexing statistics.
    """
    try:
        manager = RepositoryManager()
        indexer = CodeIndexer()
        
        # Get all repositories from cache
        repositories = []
        total_chunks = indexer.vector_store.count()
        
        # Get repository metadata from cache directory
        if settings.repo_storage_path.exists():
            for repo_dir in settings.repo_storage_path.iterdir():
                if repo_dir.is_dir() and not repo_dir.name.startswith('.'):
                    metadata_file = repo_dir / '.repo_metadata.json'
                    if metadata_file.exists():
                        import json
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        # print(metadata)
                        repositories.append(
                           RepositoryInfo(
                            repository_id=metadata.get(
                            "repository_id",
                            manager._hash_url(metadata["url"])   # fallback for old metadata
                        ),
                            repository_name=metadata["repository_name"],
                            repository_url=metadata["url"],
                            indexed_at=metadata.get("clone_time", "unknown"),
                            files_count=metadata.get("files_count", 0),
                            chunks_count=metadata.get("chunks_count", 0),
                            languages=metadata.get("languages", []),
                            metadata={
                                "branch": metadata.get("branch"),
                                "commit_sha": metadata.get("commit_sha"),
                                "last_refresh": metadata.get("last_refresh_time"),
                            }
                        )
                        )
        
        return ListRepositoriesResponse(
            repositories=repositories,
            total_count=len(repositories),
            total_chunks=total_chunks
        )
    
    except Exception as e:
        logger.error(f"List repositories failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list repositories: {str(e)}"
        )


@router.delete(
    "/repositories/{repository_id}",
    response_model=DeleteRepositoryResponse,
    summary="Delete a repository",
    description="Remove repository and its indexed vectors."
)
async def delete_repository(repository_id: str):
    """
    Delete a repository and all its indexed chunks.
    
    This removes:
    - Cached repository files
    - Indexed vectors from vector store
    - Repository metadata
    """
    logger.info(f"Delete request: {repository_id}")
    
    try:
        manager = RepositoryManager()
        
        # Find repository path by ID
        repo_path = None

        if settings.repo_storage_path.exists():
            for repo_dir in settings.repo_storage_path.iterdir():
                if not repo_dir.is_dir():
                    continue

                metadata_file = repo_dir / ".repo_metadata.json"

                if not metadata_file.exists():
                    continue

                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                repo_id = metadata.get(
                    "repository_id",
                    manager._hash_url(metadata["url"])
                )

                if repo_id == repository_id:
                    repo_path = repo_dir
                    break
                
        if not repo_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository {repository_id} not found"
            )
        
        # Delete chunks from vector store by repository_id
        indexer = CodeIndexer()
        print(indexer.vector_store.collection.peek(1)["metadatas"][0])
        chunks_deleted = indexer.vector_store.delete_by_metadata(
            where={"repository_name": metadata["repository_name"]}
        )
        
        logger.info(f"Deleted {chunks_deleted} chunks from vector store")
        
        # Delete cached repository files
        deleted = manager.delete_repository(repo_path)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete repository files"
            )
        print("Deleting repository_id:", repository_id)
        return DeleteRepositoryResponse(
            repository_id=repository_id,
            status="deleted",
            chunks_deleted=chunks_deleted
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {str(e)}"
        )


# ==============================================================================
# Health Check
# ==============================================================================

@router.get(
    "/health",
    response_model=HealthCheck,
    summary="Health check",
    description="Verify vector store, embedding model, and LLM availability."
)
async def health_check():
    """
    Check system health.
    
    Verifies:
    - Vector store connectivity
    - Embedding model availability
    - LLM provider availability
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    # Check vector store
    try:
        indexer = CodeIndexer()
        chunk_count = indexer.vector_store.count()
        health_status["vector_store"] = {
            "status": "healthy",
            "type": "ChromaDB",
            "chunks_indexed": chunk_count,
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["vector_store"] = {
            "status": "unhealthy",
            "error": str(e),
        }
    
    # Check embedding model
    try:
        from app.rag import create_embedding_model
        model = create_embedding_model()
        health_status["embedding_model"] = {
            "status": "healthy",
            "model": model.model_name,
            "dimension": model.embedding_dim,
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["embedding_model"] = {
            "status": "unhealthy",
            "error": str(e),
        }
    
    # Check LLM provider
    try:
        health_status["llm_provider"] = {
            "status": "configured",
            "provider": settings.llm_provider,
            "model": settings.llm_model,
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["llm_provider"] = {
            "status": "unhealthy",
            "error": str(e),
        }
    
    return HealthCheck(**health_status)
