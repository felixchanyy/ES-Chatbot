from fastapi import APIRouter, Request
import logging

from config import settings
from models.schemas import ChatRequest, ChatResponse, QueryMetadata
from services.query_generator import QueryGenerator, QueryGenerationError
from services.query_safety import QuerySafetyLayer, SafetyStatus
from services.es_client import ESClient
from services.context_manager import ContextManager
from services.response_summariser import ResponseSummariser
from services.chroma_client import ChromaClient

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    """
    Milestone 5: API Integration & Frontend.
    Full pipeline:
      NL question -> ES query -> safety -> execution -> result shaping -> LLM summary
    """

    session_id = payload.session_id
    logger.info(
        "chat_request_received",
        extra={
            "session_id": session_id,
        },
    )

    # 1) Generate query
    try:
        es_query = request.state.query_gen.generate(payload.message, payload.history)
    except QueryGenerationError as e:
        logger.warning(
            "Query generation failed",
            extra={
                "session_id": session_id,
                "error_type": "query_generation_error",
                "safety_status": "error"
            },
        )
        return ChatResponse(
            response="I couldn't translate that into a safe Elasticsearch query. Try rephrasing with a clearer entity/timeframe.",
            query_metadata=QueryMetadata(
                # Frontend JSON viewers (e.g., react-json-view) expect an object/array.
                # Returning null/None can render as a UI error. Use an empty object instead.
                es_query={},
                total_hits=None,
                execution_time_ms=None,
                safety_status="error",
                blocked_reason="query_generation_error",
            ),
            session_id=payload.session_id,
        )

    # 2) Safety validation / sanitisation
    validation = request.state.query_safety.validate(es_query)
    if validation.status == SafetyStatus.BLOCKED or validation.query is None:
        logger.warning(
            "Query validation failed",
            extra={
                "session_id": session_id,
                "error_type": "query_validation_error",
                "error_detail": {
                    "validation_reason": str(validation.reason),
                    "original_query": es_query,
                },
                "safety_status": validation.status.value,
            },
        )
        return ChatResponse(
            response=f"I'm sorry, I can't perform that operation. Reason: {validation.reason}",
            query_metadata=QueryMetadata(
                es_query={},
                total_hits=None,
                execution_time_ms=None,
                safety_status="blocked",
                blocked_reason=validation.reason,
            ),
            session_id=payload.session_id,
        )

    safe_query = validation.query

    # 3) Execute query
    try:
        es_resp = await request.state.es_client.search(index=settings.es_index, query=safe_query)
    except Exception as e:
        logger.error(
            "Elasticsearch execution failed",
            extra={
                "session_id": session_id,
                "error_type": "elasticsearch_execution_error",
                "error_detail": repr(e),
                "safety_status": "failed",
                "es_query": safe_query,
            }
            )
        return ChatResponse(
            response="The data store is currently unavailable or the query failed to execute. Please try again later.",
            session_id=payload.session_id,
            query_metadata=QueryMetadata(
                es_query={},
                safety_status="failed",
                blocked_reason="Elasticsearch execution failed"
            )
        )

    # 4) Shape results
    query_type = "aggregation" if (safe_query.get("aggs") or safe_query.get("aggregations") or safe_query.get("size") == 0) else "retrieval"
    shaped = request.state.context_mgr.shape_results(es_resp, query_type=query_type)

    # 5) Summarise
    answer = request.state.summariser.summarize(question=payload.message, shaped_results=shaped, query_type=query_type)

    total_hits = shaped.get("total_hits") if isinstance(shaped, dict) else None
    took_ms = shaped.get("took_ms") if isinstance(shaped, dict) else None

    return ChatResponse(
        response=answer,
        query_metadata=QueryMetadata(
            es_query=safe_query,
            es_response=es_resp,
            total_hits=total_hits,
            execution_time_ms=took_ms,
            safety_status=validation.status.value,
            blocked_reason=validation.reason if validation.status == SafetyStatus.MODIFIED else None,
        ),
        session_id=payload.session_id,
    )