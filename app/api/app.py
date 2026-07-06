"""
FastAPI application setup.

Main application entry point.

Includes:
- Global exception handler (structured error responses for all unhandled errors)
- Request/response logging middleware (method, path, status, duration)
- Rate limiting middleware (token bucket, per IP)
- CORS middleware
"""
import time
import uuid
from collections import defaultdict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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


# ==============================================================================
# 1. Global Exception Handler
# ==============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch all unhandled exceptions and return a structured error response.

    Logs the full traceback so it shows up in app.log, but returns a clean
    JSON body to the client so internal details are not leaked.
    """
    import traceback

    error_id = str(uuid.uuid4())[:8]  # Short ID for correlating logs to responses

    logger.error(
        f"[{error_id}] Unhandled exception on {request.method} {request.url.path}\n"
        + traceback.format_exc()
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
            "error_id": error_id,  # Client can report this for debugging
        },
    )


@app.exception_handler(TimeoutError)
async def timeout_exception_handler(request: Request, exc: TimeoutError) -> JSONResponse:
    """Handle LLM / network timeout errors specifically."""
    logger.warning(f"Timeout on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={
            "error": "timeout",
            "message": "The request timed out. The LLM or a downstream service took too long.",
        },
    )


# ==============================================================================
# 2. Request / Response Logging Middleware
# ==============================================================================

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Log every incoming request and its response.

    Logs: method, path, query string, client IP, response status, and duration.
    Attaches a unique request ID to each request for tracing.
    """
    request_id = str(uuid.uuid4())[:8]
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path
    query = f"?{request.url.query}" if request.url.query else ""

    logger.info(
        f"[{request_id}] → {request.method} {path}{query} "
        f"from {client_ip}"
    )

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        duration = (time.perf_counter() - start) * 1000
        logger.error(
            f"[{request_id}] ✗ {request.method} {path} "
            f"FAILED after {duration:.1f}ms — {exc}"
        )
        raise

    duration = (time.perf_counter() - start) * 1000
    logger.info(
        f"[{request_id}] ← {response.status_code} "
        f"{request.method} {path} "
        f"({duration:.1f}ms)"
    )

    # Pass the request ID back so clients can reference it
    response.headers["X-Request-ID"] = request_id
    return response


# ==============================================================================
# 3. Rate Limiting Middleware (token bucket, per IP)
# ==============================================================================

# Limits per endpoint category (requests per minute)
_RATE_LIMITS = {
    "/repositories/index": 3,   # Indexing is expensive — 3 req/min per IP
    "/chat":               20,  # Chat is lighter — 20 req/min per IP
    "default":             60,  # Everything else — 60 req/min per IP
}

# State: {ip: {path_key: [timestamp, ...]}}
_rate_limit_state: dict = defaultdict(lambda: defaultdict(list))


def _get_rate_limit(path: str) -> int:
    """Return the requests-per-minute limit for a given path."""
    for key, limit in _RATE_LIMITS.items():
        if key != "default" and key in path:
            return limit
    return _RATE_LIMITS["default"]


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Simple sliding-window rate limiter per client IP.

    Tracks request timestamps in a 60-second window.
    Returns 429 if the client exceeds the limit for that endpoint category.
    """
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path
    limit = _get_rate_limit(path)
    now = time.time()
    window = 60.0  # 1-minute sliding window

    # Clean up timestamps outside the window
    timestamps = _rate_limit_state[client_ip][path]
    _rate_limit_state[client_ip][path] = [t for t in timestamps if now - t < window]

    if len(_rate_limit_state[client_ip][path]) >= limit:
        logger.warning(
            f"Rate limit exceeded: {client_ip} on {path} "
            f"({limit} req/min)"
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limit_exceeded",
                "message": f"Too many requests. Limit is {limit} requests per minute for this endpoint.",
                "retry_after_seconds": 60,
            },
            headers={"Retry-After": "60"},
        )

    _rate_limit_state[client_ip][path].append(now)
    return await call_next(request)


# ==============================================================================
# CORS Middleware
# ==============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["Code Q&A"])


# ==============================================================================
# Lifecycle events
# ==============================================================================

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
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "healthy"}
