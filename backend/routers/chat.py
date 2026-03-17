from __future__ import annotations

import logging

from fastapi import APIRouter

from models.schemas import ChatRequest, ChatResponse, QueryMetadata
from services.react_agent import ReActGDELTAgentService

router = APIRouter()
logger = logging.getLogger(__name__)
agent_service = ReActGDELTAgentService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    logger.info("chat_request_received", extra={"session_id": request.session_id})

    history = [item.model_dump() for item in request.history]

    result = await agent_service.run(
        question=request.message,
        history=history,
    )

    return ChatResponse(
        response=result["response"],
        query_metadata=QueryMetadata(
            es_query=result["es_query"],
            total_hits=result["total_hits"],
            execution_time_ms=result["execution_time_ms"],
            safety_status=result["safety_status"],
            blocked_reason=result["blocked_reason"],
            attempts=result["attempts"],
            validator_reason=result["validator_reason"],
            stage_trace=result["stage_trace"],
        ),
        session_id=request.session_id,
    )