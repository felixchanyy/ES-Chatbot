# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chat, index

app = FastAPI(
    title="GKG OSINT Chatbot API",
    version="1.0.0",
    description="Natural language query interface for the GDELT GKG Elasticsearch index"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://frontend:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1")
app.include_router(index.router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    # Return ES connectivity status and LLM reachability
    return {"status": "ok"}