import logging

import requests
from elasticsearch import Elasticsearch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from routers import chat, index
from services.logging_config import setup_logging
from services.schema_store import get_schema_store

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


@app.on_event("startup")
async def startup_event() -> None:
    try:
        schema_store = get_schema_store()
        await schema_store.ensure_schema_collection_synced(force=False)
    except Exception:
        logger.exception("startup_schema_sync_failed")


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
        r = requests.get(f"{settings.llm_base_url}/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _check_chromadb() -> bool:
    """Check Chroma health.

    Try v2 first because the Python HttpClient expects v2-capable servers.
    Fall back to v1 for compatibility during debugging / transitional states.
    """
    base = f"http://{settings.chroma_host}:{settings.chroma_port}"
    try:
        r = requests.get(f"{base}/api/v2/heartbeat", timeout=5)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    return False

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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    session_id = getattr(request.state, "session_id", None)
    logger.exception(
        "Unhandled exception",
        extra={
            "session_id": session_id,
            "error_type": "unhandled_exception",
            "error_detail": repr(exc),
            "phase": "global_handling",
        },
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )