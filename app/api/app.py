"""
FastAPI application setup.

Main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import settings, logger
from app.api.routes import router


# Create FastAPI app
app = FastAPI(
    title="Code Q&A Assistant",
    description="RAG-based code question answering system using semantic search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["Code Q&A"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Code Q&A Assistant API")
    logger.info(f"Vector store: {settings.chroma_persist_directory}")
    logger.info(f"Embedding model: {settings.embedding_model}")
    logger.info(f"LLM provider: {settings.llm_provider}")
    logger.info(f"LLM model: {settings.llm_model}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Code Q&A Assistant API")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Code Q&A Assistant API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "healthy"}
