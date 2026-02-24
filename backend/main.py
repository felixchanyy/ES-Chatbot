from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import Elasticsearch, logger
import requests
import logging 
from fastapi import FastAPI, Request 
from fastapi.responses import JSONResponse
from config import settings
from routers import chat, index

logger = logging.getLogger("uvicorn.error") # This connects to the FastAPI/Uvicorn terminal output

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

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log the real error for the developer to see in the terminal
    logger.error(f"Global crash caught: {exc}")
    
    # Return a generic, safe message to the user
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."}
    )