from fastapi import APIRouter
import logging
logger = logging.getLogger(__name__)
# import settings

from models.schemas import ChatRequest, ChatResponse, QueryMetadata
from services.query_generator import QueryGenerator, QueryGenerationError
from services.query_safety import QuerySafetyLayer, SafetyStatus
from services.es_client import ESClient
from services.context_manager import ContextManager
from services.response_summariser import ResponseSummariser
from config import settings

router = APIRouter()
query_gen = QueryGenerator()
query_safety = QuerySafetyLayer()
es = ESClient()
context_mgr = ContextManager(max_docs=query_safety.max_result_docs)
summariser = ResponseSummariser()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Milestone 4:
    Full pipeline:
      NL question -> ES query -> safety -> execution -> result shaping -> LLM summary
    """

    # 1) Generate query
    try:
        es_query = query_gen.generate(request.message, request.history)
    except QueryGenerationError:
        return ChatResponse(
            response="I couldn't translate that into a safe Elasticsearch query. Try rephrasing with a clearer entity/timeframe.",
            query_metadata=QueryMetadata(
                # Frontend JSON viewers (e.g., react-json-view) expect an object/array.
                # Returning null/None can render as a UI error. Use an empty object instead.
                es_query={},
                total_hits=None,
                execution_time_ms=None,
                safety_status="blocked",
                blocked_reason="query_generation_error",
            ),
            session_id=request.session_id,
        )

    # 2) Safety validation / sanitisation
    validation = query_safety.validate(es_query)
    if validation.status == SafetyStatus.BLOCKED or validation.query is None:
        return ChatResponse(
            response=f"I'm sorry, I can't perform that operation. Reason: {validation.reason}",
            query_metadata=QueryMetadata(
                es_query={},
                total_hits=None,
                execution_time_ms=None,
                safety_status="blocked",
                blocked_reason=validation.reason,
            ),
            session_id=request.session_id,
        )

    safe_query = validation.query

    # 3) Execute query
    try:
        es_resp = await es.search(index=settings.es_index, query=safe_query)
    except Exception as e:
        logger.exception("Elasticsearch execution failed: %s", e)
        return ChatResponse(
            response="The data store is currently unavailable or the query failed to execute. Please try again later.",
            session_id=request.session_id,
            query_metadata=QueryMetadata(
                es_query={},
                safety_status="blocked", # Changed from "error" to fit the allowed Literals
                blocked_reason="Elasticsearch execution failed"
            )
        )

    # 4) Shape results
    query_type = "aggregation" if (safe_query.get("aggs") or safe_query.get("aggregations") or safe_query.get("size") == 0) else "retrieval"
    shaped = context_mgr.shape_results(es_resp, query_type=query_type)

    # 5) Summarise
    answer = summariser.summarize(question=request.message, shaped_results=shaped, query_type=query_type)

    total_hits = shaped.get("total_hits") if isinstance(shaped, dict) else None
    took_ms = shaped.get("took_ms") if isinstance(shaped, dict) else None

    return ChatResponse(
        response=answer,
        query_metadata=QueryMetadata(
            es_query=safe_query,
            total_hits=total_hits,
            execution_time_ms=took_ms,
            safety_status=validation.status.value,
            blocked_reason=validation.reason if validation.status == SafetyStatus.MODIFIED else None,
        ),
        session_id=request.session_id,
    )