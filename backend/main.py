import logging
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.ingestion import router as ingestion_router
from backend.api.routes.search import router as search_router
from backend.api.routes.chat import router as chat_router
from backend.api.routes.recommendation import router as recommendation_router
from backend.api.routes.literature import router as literature_router
from backend.api.routes.comparison import router as comparison_router
from backend.api.routes.auth import router as auth_router
from backend.middleware.error_handler import global_exception_handler
from backend.retrieval.bm25_corpus import ensure_corpus_index
from backend.retrieval.reranker import get_reranker
from backend.utils.config import get_settings
from backend.utils.logger import setup_logger

logger = setup_logger()
logging.getLogger("backend.ingestion").setLevel(logging.INFO)
logging.getLogger("backend.retrieval").setLevel(logging.INFO)

app = FastAPI(title="AI Research Assistant Platform", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(Exception, global_exception_handler)
app.include_router(auth_router, prefix="/api/v1")
app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(recommendation_router, prefix="/api/v1")
app.include_router(literature_router, prefix="/api/v1")
app.include_router(comparison_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root():
    """Avoid a bare 404 when someone opens the API host in a browser."""
    return {
        "service": "ai-research-assistant-backend",
        "docs": "/docs",
        "health": "/health",
        "api_prefix": "/api/v1",
        "examples": {
            "chat": "POST /api/v1/chat/",
            "search": "POST /api/v1/search/",
        },
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ai-research-assistant-backend"}


def _warm_retrieval_background() -> None:
    """BM25 + reranker can take many minutes on CPU; run off the ASGI startup path so
    Uvicorn binds and /health works immediately while warmup continues.
    """
    settings = get_settings()
    if settings.retrieval_use_corpus_bm25:
        try:
            ensure_corpus_index()
            logger.info("Startup warmup: corpus BM25 ready")
        except Exception:
            logger.exception("Startup warmup failed for corpus BM25")
    try:
        get_reranker()
        logger.info("Startup warmup: reranker ready")
    except Exception:
        logger.exception("Startup warmup failed for reranker")


@app.on_event("startup")
def warm_retrieval_components() -> None:
    threading.Thread(
        target=_warm_retrieval_background,
        name="retrieval-warmup",
        daemon=True,
    ).start()
    logger.info("Startup warmup scheduled in background (BM25 + reranker); API is accepting requests")
