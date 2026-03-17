from __future__ import annotations

import logging

from fastapi import APIRouter

from models.schemas import ChatRequest, ChatResponse, QueryMetadata
from services.react_agent import ReActGDELTAgentService

router = APIRouter()
logger = logging.getLogger(__name__)
agent_service = ReActGDELTAgentService()
from services.context_manager import ContextManager
from services.es_client import es_client
from services.query_generator import QueryGenerationError, QueryGenerator
from services.query_safety import QuerySafetyLayer, SafetyStatus
from services.response_summariser import ResponseSummariser

router = APIRouter()
logger = logging.getLogger(__name__)
query_gen = QueryGenerator()
query_safety = QuerySafetyLayer()
context_mgr = ContextManager(max_docs=query_safety.max_result_docs)
summariser = ResponseSummariser()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    logger.info("chat_request_received", extra={"session_id": request.session_id})
    history = [item.model_dump() for item in request.history]
    result = await agent_service.run(question=request.message, history=history)
    observations: List[str] = []
    attempts: List[Dict[str, Any]] = []
    seen_queries: set[str] = set()
    final_attempt: Dict[str, Any] | None = None
    last_validation_reason: str | None = None

    for attempt_no in range(1, settings.query_max_attempts + 1):
        previous_queries = [attempt["query"] for attempt in attempts]

        try:
            es_query = await query_gen.generate(
                question=request.message,
                history=history,
                previous_queries=previous_queries,
                observations=observations,
            )
        except QueryGenerationError as exc:
            logger.warning(
                "query_generation_failed",
                extra={
                    "session_id": session_id,
                    "error_type": "query_generation_error",
                    "error_detail": str(exc),
                    "safety_status": "error",
                },
            )
            observations.append(f"Attempt {attempt_no}: query generation failed - {exc}")
            if attempts:
                break
            return ChatResponse(
                response="I couldn't translate that into a safe Elasticsearch query. Try rephrasing with a clearer entity or timeframe.",
                query_metadata=QueryMetadata(
                    es_query={},
                    total_hits=None,
                    execution_time_ms=None,
                    safety_status="error",
                    blocked_reason="query_generation_error",
                ),
                session_id=session_id,
            )
        except Exception as exc:
            logger.exception(
                "query_generation_runtime_failure",
                extra={
                    "session_id": session_id,
                    "error_type": "query_generation_runtime_error",
                    "error_detail": repr(exc),
                    "safety_status": "error",
                },
            )
            observations.append(f"Attempt {attempt_no}: query generation runtime failure - {exc}")
            if attempts:
                break
            return ChatResponse(
                response="Query generation failed because the retrieval or LLM service was unavailable.",
                query_metadata=QueryMetadata(
                    es_query={},
                    total_hits=None,
                    execution_time_ms=None,
                    safety_status="error",
                    blocked_reason="query_generation_runtime_error",
                ),
                session_id=session_id,
            )

        validation = query_safety.validate(es_query)
        last_validation_reason = validation.reason
        if validation.status == SafetyStatus.BLOCKED or validation.query is None:
            logger.warning(
                "query_validation_failed",
                extra={
                    "session_id": session_id,
                    "error_type": "query_validation_error",
                    "error_detail": validation.reason,
                    "safety_status": validation.status.value,
                },
            )
            observations.append(f"Attempt {attempt_no}: query blocked by safety layer - {validation.reason}")
            if attempts:
                break
            return ChatResponse(
                response=f"I'm sorry, I can't perform that operation. Reason: {validation.reason}",
                query_metadata=QueryMetadata(
                    es_query={},
                    total_hits=None,
                    execution_time_ms=None,
                    safety_status="blocked",
                    blocked_reason=validation.reason,
                ),
                session_id=session_id,
            )

        safe_query = validation.query
        signature = _canonical_query(safe_query)
        if signature in seen_queries:
            observations.append(f"Attempt {attempt_no}: duplicate follow-up query was prevented.")
            break
        seen_queries.add(signature)

        query_type = _infer_query_type(safe_query)

        try:
            es_resp = await es_client.search(index=settings.es_index, query=safe_query)
        except Exception as exc:
            logger.exception(
                "elasticsearch_execution_failed",
                extra={
                    "session_id": session_id,
                    "error_type": "elasticsearch_execution_error",
                    "error_detail": repr(exc),
                    "safety_status": "failed",
                    "es_query": safe_query,
                },
            )
            observations.append(f"Attempt {attempt_no}: Elasticsearch execution failed - {exc}")
            if attempts:
                break
            return ChatResponse(
                response="The data store is currently unavailable or the query failed to execute. Please try again later.",
                session_id=session_id,
                query_metadata=QueryMetadata(
                    es_query={},
                    total_hits=None,
                    execution_time_ms=None,
                    safety_status="failed",
                    blocked_reason="elasticsearch_execution_failed",
                ),
            )

        shaped = context_mgr.shape_results(es_resp, query_type=query_type)
        observation = _build_stage_observation(
            attempt_no=attempt_no,
            query=safe_query,
            shaped=shaped,
            query_type=query_type,
        )
        observations.append(observation)

        attempt_record = {
            "attempt": attempt_no,
            "query": safe_query,
            "query_type": query_type,
            "shaped": shaped,
            "safety_status": validation.status.value,
            "safety_reason": validation.reason,
            "observation": observation,
        }
        attempts.append(attempt_record)

        if _has_useful_results(shaped, query_type):
            final_attempt = attempt_record
            break

    if final_attempt is None and attempts:
        final_attempt = max(attempts, key=_score_attempt)

    if final_attempt is None:
        return ChatResponse(
            response="I could not obtain a reliable result after several attempts.",
            query_metadata=QueryMetadata(
                es_query={},
                total_hits=None,
                execution_time_ms=None,
                safety_status="error",
                blocked_reason=last_validation_reason or "no_executable_query",
            ),
            session_id=session_id,
        )

    stage_trace = [
        {
            "attempt": item["attempt"],
            "query_type": item["query_type"],
            "safety_status": item["safety_status"],
            "observation": item["observation"],
            "query": item["query"],
        }
        for item in attempts
    ]

    answer = summariser.summarize(
        question=request.message,
        shaped_results=final_attempt["shaped"],
        query_type=final_attempt["query_type"],
        stage_trace=stage_trace,
    )

    shaped = final_attempt["shaped"] if isinstance(final_attempt["shaped"], dict) else {}
    total_hits = shaped.get("total_hits")
    took_ms = shaped.get("took_ms")

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