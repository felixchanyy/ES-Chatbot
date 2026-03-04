import logging

from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import Elasticsearch
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from config import settings
from routers import chat, index
from services.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GKG OSINT Chatbot API",
    version="1.0.0",
    description="Natural language query interface for the GDELT GKG Elasticsearch index",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://frontend:8501", "http://localhost:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1")
app.include_router(index.router, prefix="/api/v1")


# -------------------------
# Health Check Helpers
# -------------------------

def _check_elasticsearch() -> bool:
    try:
        es = Elasticsearch(
            settings.es_host,
            basic_auth=(settings.es_username, settings.es_password),
            verify_certs=settings.es_verify_ssl,
            request_timeout=5,
        )
        return bool(es.ping())
    except Exception:
        return False


def _check_llm() -> bool:
    try:
        r = requests.get(
            f"{settings.llm_base_url}/models",
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


def _check_chromadb() -> bool:
    try:
        r = requests.get(
            f"http://{settings.chroma_host}:{settings.chroma_port}/api/v2/heartbeat",
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


# -------------------------
# Health Endpoint
# -------------------------

@app.get("/health")
def health_check():
    es_ok = _check_elasticsearch()
    llm_ok = _check_llm()
    chroma_ok = _check_chromadb()

    overall_ok = es_ok and llm_ok and chroma_ok
    return {
        "status": "ok" if overall_ok else "degraded",
        "elasticsearch": es_ok,
        "llm": llm_ok,
        "chromadb": chroma_ok,
    }

# -------------------------
# Error Handling
# -------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # This will catch any unhandled exceptions in the application and log them with structured logging
    # This should be the last resort for error handling, as specific exceptions should ideally be caught and handled in their respective routes or services for better granularity and user feedback.
    session_id = getattr(request.state, "session_id", None)
    logger.exception(
        f"Unhandled exception: {exc}",
        extra={
            "session_id": session_id,
            "error_type": "undhandled_exception",
            "error_detail": repr(exc),
            "phase": "global_handling",
        }
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )