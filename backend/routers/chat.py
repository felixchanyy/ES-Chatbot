from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter

from config import settings
from models.schemas import ChatRequest, ChatResponse, QueryMetadata
from services.context_manager import ContextManager
from services.es_client import ESClient
from services.query_generator import QueryGenerationError, QueryGenerator
from services.query_safety import QuerySafetyLayer, SafetyStatus
from services.response_summariser import ResponseSummariser
from services.result_validator import ResultValidator

validator = ResultValidator()
router = APIRouter()
logger = logging.getLogger(__name__)
query_gen = QueryGenerator()
query_safety = QuerySafetyLayer()
es = ESClient()
context_mgr = ContextManager(max_docs=query_safety.max_result_docs)
summariser = ResponseSummariser()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Multi-stage chat pipeline.

    Updated behaviour:
    - keeps state across attempts
    - retries when a query returns no useful results
    - avoids duplicate follow-up queries
    - falls back safely if a stage fails
    """

    session_id = request.session_id
    logger.info("chat_request_received", extra={"session_id": session_id})

    history = [item.model_dump() for item in request.history]
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
            es_resp = await es.search(index=settings.es_index, query=safe_query)
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

        validation_outcome = None

        if query_type == "aggregation":
            validation_outcome = validator.validate_aggregation(request.message, shaped)

            if validation_outcome.valid_items:
                # optionally overwrite shaped results so downstream summary uses validated buckets
                first_agg_name = next(iter(shaped["aggregations"].keys()))
                shaped["aggregations"][first_agg_name] = validation_outcome.valid_items

            attempt_record["validation_outcome"] = {
                "is_sufficient": validation_outcome.is_sufficient,
                "reason": validation_outcome.reason,
                "invalid_items": validation_outcome.invalid_items,
            }

            if validation_outcome.is_sufficient:
                final_attempt = attempt_record
                break

            observations.append(
                f"Attempt {attempt_no}: result insufficient. "
                f"Reason={validation_outcome.reason}. "
                f"Rejected buckets={[x.get('key') for x in validation_outcome.invalid_items[:10]]}. "
                f"Need more valid entities."
            )
            continue

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
        response=answer,
        query_metadata=QueryMetadata(
            es_query=final_attempt["query"],
            total_hits=total_hits,
            execution_time_ms=took_ms,
            safety_status=final_attempt["safety_status"],
            blocked_reason=final_attempt["safety_reason"],
        ),
        session_id=session_id,
    )


def _canonical_query(query: Dict[str, Any]) -> str:
    return json.dumps(query, sort_keys=True, ensure_ascii=False)


def _infer_query_type(query: Dict[str, Any]) -> str:
    return "aggregation" if bool(query.get("aggs") or query.get("aggregations")) else "retrieval"


def _has_useful_results(shaped: Dict[str, Any], query_type: str) -> bool:
    if not isinstance(shaped, dict) or shaped.get("error"):
        return False

    if query_type == "aggregation":
        return _has_non_empty_buckets(shaped.get("aggregations"))

    docs = shaped.get("documents") or []
    total_hits = shaped.get("total_hits")
    return bool(docs) or (isinstance(total_hits, int) and total_hits > 0)


def _has_non_empty_buckets(node: Any) -> bool:
    if isinstance(node, list):
        return len(node) > 0
    if isinstance(node, dict):
        return any(_has_non_empty_buckets(value) for value in node.values())
    return False


def _score_attempt(attempt: Dict[str, Any]) -> int:
    shaped = attempt.get("shaped") if isinstance(attempt.get("shaped"), dict) else {}
    if attempt.get("query_type") == "aggregation":
        return 100 if _has_non_empty_buckets(shaped.get("aggregations")) else 0
    docs = shaped.get("documents") or []
    total_hits = shaped.get("total_hits") or 0
    return len(docs) * 10 + int(total_hits)


def _build_stage_observation(
    *,
    attempt_no: int,
    query: Dict[str, Any],
    shaped: Dict[str, Any],
    query_type: str,
    validation_reason: str | None = None,
    invalid_keys: list[str] | None = None,
) -> str:
    if not isinstance(shaped, dict):
        return f"Attempt {attempt_no}: result shaping failed."

    if query_type == "aggregation":
        if validation_reason:
            invalid_part = ""
            if invalid_keys:
                invalid_part = f" Invalid buckets: {', '.join(invalid_keys[:10])}."
            return (
                f"Attempt {attempt_no}: aggregation returned buckets but result is insufficient. "
                f"{validation_reason}.{invalid_part} "
                "Need a reformulated query to get enough valid entities."
            )

        has_buckets = _has_non_empty_buckets(shaped.get("aggregations"))
        if has_buckets:
            return f"Attempt {attempt_no}: aggregation query returned non-empty buckets."
        return f"Attempt {attempt_no}: aggregation query returned no buckets; consider broader filters or a wider time range."

    total_hits = shaped.get("total_hits")
    docs = shaped.get("documents") or []
    if docs:
        return f"Attempt {attempt_no}: retrieval query returned {len(docs)} documents and about {total_hits} total hits."
    if isinstance(total_hits, int) and total_hits > 0:
        return f"Attempt {attempt_no}: query matched about {total_hits} hits but returned no document previews."
    return f"Attempt {attempt_no}: query returned zero useful results; simplify the query or broaden the time window."