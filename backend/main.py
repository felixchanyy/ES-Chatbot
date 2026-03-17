import logging

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from routers import chat, index
from services.schema_store import get_schema_store
from services.es_client import es_client

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GKG OSINT Chatbot API",
    version="2.0.0",
    description="LangChain-agent-powered natural language interface for the GDELT Elasticsearch index",
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

def _check_llm() -> bool:
    try:
        response = requests.get(f"{settings.llm_base_url}/models", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def _check_chromadb() -> bool:
    try:
        response = requests.get(
            f"http://{settings.chroma_host}:{settings.chroma_port}/api/v2/heartbeat",
            timeout=5,
        )
        return response.status_code == 200
    except Exception:
        return False


@app.get("/health")
def health_check() -> dict:
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
    logger.exception(
        "Unhandled exception",
        extra={"path": request.url.path, "error_detail": repr(exc)},
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )