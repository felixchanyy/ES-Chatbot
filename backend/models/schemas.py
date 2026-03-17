from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(..., description="UUID identifying the chat session")
    history: List[HistoryItem] = Field(default_factory=list, max_length=30)


class QueryMetadata(BaseModel):
    es_query: Optional[Dict[str, Any]] = None
    total_hits: Optional[int] = None
    execution_time_ms: Optional[int] = None
    safety_status: Literal["allowed", "blocked", "modified", "failed", "error"]
    blocked_reason: Optional[str] = None
    attempts: int = 0
    validator_reason: Optional[str] = None
    stage_trace: List[Dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    query_metadata: QueryMetadata
    session_id: str


class SourceCount(BaseModel):
    source: str
    count: int


class IndexStats(BaseModel):
    total_documents: int
    index_size_bytes: int
    earliest_date: Optional[str] = None
    latest_date: Optional[str] = None
    top_sources: List[SourceCount]