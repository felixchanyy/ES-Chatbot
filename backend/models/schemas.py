# backend/models/schemas.py

from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Optional

class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(..., description="UUID identifying the chat session")
    history: List[HistoryItem] = Field(default_factory=list, max_length=20)

class QueryMetadata(BaseModel):
    es_query: Optional[Dict] = Field(None, description="The ES query that was executed")
    total_hits: Optional[int] = None
    execution_time_ms: Optional[int] = None
    safety_status: Literal["allowed", "blocked", "modified"]
    blocked_reason: Optional[str] = None

class ChatResponse(BaseModel):
    response: str = Field(..., description="Natural language response from the LLM")
    query_metadata: QueryMetadata
    session_id: str

class IndexStats(BaseModel):
    total_documents: int
    index_size_bytes: int
    earliest_date: str
    latest_date: str
    top_sources: List[Dict]