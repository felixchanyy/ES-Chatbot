# backend/routers/index.py

from fastapi import APIRouter, Request
from models.schemas import IndexStats
from services.es_client import ESClient

router = APIRouter()

@router.get("/index/stats", response_model=IndexStats)
async def get_index_stats(request: Request):
    """Return summary statistics about the GKG index."""
    es_client = request.state.es_client
    stats = await es_client.get_index_stats()
    return stats