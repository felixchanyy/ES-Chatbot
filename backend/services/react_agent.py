from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from config import settings
from services.context_manager import ContextManager
from services.query_generator import QueryGenerator, QueryGenerationError
from services.query_safety import QuerySafetyLayer, SafetyStatus
from services.response_summariser import ResponseSummariser
from services.result_validator import ResultValidator, ValidationOutcome

logger = logging.getLogger(__name__)


@dataclass
class AgentRunState:
    question: str
    history: List[Dict[str, Any]]
    stage_trace: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    previous_queries: List[Dict[str, Any]] = field(default_factory=list)
    latest_query: Optional[Dict[str, Any]] = None
    latest_raw_results: Optional[Dict[str, Any]] = None
    latest_shaped_results: Optional[Dict[str, Any]] = None
    latest_query_type: Optional[str] = None
    latest_safety_status: str = SafetyStatus.ALLOWED.value
    latest_safety_reason: Optional[str] = None
    last_validation: Optional[ValidationOutcome] = None
    hard_stop: bool = False
    hard_stop_reason: Optional[str] = None
    execution_failures: int = 0


class ReActGDELTAgentService:
    def __init__(self) -> None:
        self.model = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
        self.query_generator = QueryGenerator()
        self.safety = QuerySafetyLayer()
        self.context_manager = ContextManager(max_docs=settings.max_result_docs)
        self.validator = ResultValidator()
        self.summariser = ResponseSummariser()

        self.sync_es = Elasticsearch(
            hosts=[settings.es_host],
            basic_auth=(settings.es_username, settings.es_password),
            verify_certs=settings.es_verify_ssl,
            request_timeout=settings.es_request_timeout_seconds,
        )

    async def run(self, *, question: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        state = AgentRunState(question=question, history=history)
        tools = self._build_tools(state)

        agent = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=self._system_prompt(),
        )

        agent_result = None
        try:
            agent_result = await asyncio.to_thread(
                agent.invoke,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"User question: {question}\n"
                                "Use the tools. If validation passes, stop. "
                                "If execution repeatedly fails or retry budget is exhausted, stop."
                            ),
                        }
                    ]
                },
                {"recursion_limit": settings.agent_max_iterations},
            )
        except GraphRecursionError as exc:
            logger.exception("agent_recursion_limit_reached")
            state.hard_stop = True
            state.hard_stop_reason = f"agent_recursion_limit_reached: {exc}"
            state.stage_trace.append(
                {
                    "stage": "agent",
                    "status": "hard_stop",
                    "reason": state.hard_stop_reason,
                }
            )

        final_validation = state.last_validation
        final_results = self._validated_final_results(state)

        answer = await self.summariser.summarize(
            question=question,
            shaped_results=final_results,
            stage_trace=state.stage_trace,
        )

        if state.hard_stop and not final_results:
            answer = (
                "I could not complete the query successfully because the generated Elasticsearch queries "
                "kept failing validation or execution before a valid answer was obtained."
            )

        return {
            "response": answer,
            "es_query": state.latest_query,
            "total_hits": (state.latest_shaped_results or {}).get("total_hits"),
            "execution_time_ms": (state.latest_shaped_results or {}).get("took_ms"),
            "safety_status": state.latest_safety_status,
            "blocked_reason": state.latest_safety_reason or state.hard_stop_reason,
            "attempts": len(state.previous_queries),
            "validator_reason": final_validation.reason if final_validation else state.hard_stop_reason,
            "stage_trace": state.stage_trace,
            "agent_result": agent_result,
        }

    def _system_prompt(self) -> str:
        return """
You are a GDELT Elasticsearch ReAct agent.

Use tools in this order:
1. build_es_query
2. execute_latest_query
3. validate_latest_results
4. If validation fails and retries remain, build_es_query again with refinement guidance.
5. If validation passes, stop.
6. If a tool says hard stop, retry limit reached, or repeated execution failure, stop immediately.

Important:
- Do not loop forever.
- If execution fails more than once, stop instead of repeatedly retrying the same broken path.
- Filtering invalid buckets is not success unless requested count is still satisfied.
""".strip()

    def _validated_final_results(self, state: AgentRunState) -> Dict[str, Any]:
        if not state.latest_shaped_results:
            return {}

        validation = state.last_validation
        if not validation:
            return state.latest_shaped_results

        aggs = state.latest_shaped_results.get("aggregations") or {}
        if not isinstance(aggs, dict) or not aggs:
            return state.latest_shaped_results

        first_key = next(iter(aggs.keys()))
        final_results = dict(state.latest_shaped_results)
        final_aggs = dict(aggs)

        # Always replace the agg output with validated items, even if fewer than requested.
        final_aggs[first_key] = validation.valid_items
        final_results["aggregations"] = final_aggs

        # Preserve failure metadata so the summariser can say the result is incomplete.
        final_results["validation"] = {
            "passed": validation.passed,
            "reason": validation.reason,
            "requested_count": validation.requested_count,
            "valid_count": validation.valid_count,
        }
        return final_results

    def _build_tools(self, state: AgentRunState) -> List[Any]:
        service = self

        @tool
        def build_es_query(refinement_hint: str = "") -> str:
            """Build or rebuild a safe read-only Elasticsearch query."""
            if state.hard_stop:
                return json.dumps(
                    {"status": "hard_stop", "reason": state.hard_stop_reason},
                    ensure_ascii=False,
                )

            if len(state.previous_queries) >= settings.query_max_attempts:
                state.hard_stop = True
                state.hard_stop_reason = "retry_limit_reached"
                return json.dumps(
                    {"status": "retry_limit_reached", "attempts": len(state.previous_queries)},
                    ensure_ascii=False,
                )

            try:
                query = asyncio.run(
                    service.query_generator.build_query(
                        question=state.question,
                        history=state.history,
                        observations=state.observations,
                        previous_queries=state.previous_queries,
                        refinement_hint=refinement_hint,
                    )
                )
            except QueryGenerationError as exc:
                state.hard_stop = True
                state.hard_stop_reason = f"query_generation_error: {exc}"
                return json.dumps({"status": "hard_stop", "reason": state.hard_stop_reason}, ensure_ascii=False)
            except Exception as exc:
                logger.exception("build_es_query_failed")
                state.hard_stop = True
                state.hard_stop_reason = f"query_build_exception: {repr(exc)}"
                return json.dumps({"status": "hard_stop", "reason": state.hard_stop_reason}, ensure_ascii=False)

            safety = service.safety.validate(query)
            state.latest_safety_status = safety.status.value
            state.latest_safety_reason = safety.reason

            if safety.status == SafetyStatus.BLOCKED:
                state.observations.append(f"safety_blocked: {safety.reason}")
                state.stage_trace.append(
                    {
                        "stage": "safety",
                        "status": "blocked",
                        "reason": safety.reason,
                    }
                )
                state.hard_stop = True
                state.hard_stop_reason = f"safety_blocked: {safety.reason}"
                return json.dumps(
                    {"status": "hard_stop", "reason": state.hard_stop_reason},
                    ensure_ascii=False,
                )

            safe_query = safety.query or query
            state.latest_query = safe_query
            state.previous_queries.append(safe_query)

            state.stage_trace.append(
                {
                    "stage": "build_query",
                    "attempt": len(state.previous_queries),
                    "status": safety.status.value,
                    "reason": safety.reason,
                    "query": safe_query,
                }
            )
            return json.dumps(
                {
                    "status": "ok",
                    "attempt": len(state.previous_queries),
                    "query": safe_query,
                    "safety_status": safety.status.value,
                },
                ensure_ascii=False,
            )

        @tool
        def execute_latest_query() -> str:
            """Execute the latest query."""
            if state.hard_stop:
                return json.dumps(
                    {"status": "hard_stop", "reason": state.hard_stop_reason},
                    ensure_ascii=False,
                )

            if not state.latest_query:
                state.hard_stop = True
                state.hard_stop_reason = "no_query_built"
                return json.dumps({"status": "hard_stop", "reason": state.hard_stop_reason}, ensure_ascii=False)

            try:
                response = service.sync_es.search(index=settings.es_index, body=state.latest_query)
                raw = response.body if hasattr(response, "body") else response
            except Exception as exc:
                logger.exception("execute_latest_query_failed")
                state.execution_failures += 1
                err = repr(exc)
                state.observations.append(f"execution_failed: {err}")
                state.stage_trace.append(
                    {
                        "stage": "execute",
                        "attempt": len(state.previous_queries),
                        "status": "error",
                        "reason": err,
                    }
                )

                if state.execution_failures >= 2:
                    state.hard_stop = True
                    state.hard_stop_reason = f"repeated_execution_failure: {err}"
                    return json.dumps(
                        {"status": "hard_stop", "reason": state.hard_stop_reason},
                        ensure_ascii=False,
                    )

                return json.dumps(
                    {
                        "status": "execution_failed",
                        "reason": err,
                        "message": "Refine the query and try once more.",
                    },
                    ensure_ascii=False,
                )

            state.latest_raw_results = raw
            query_type = "aggregation" if (state.latest_query.get("aggs") or state.latest_query.get("aggregations")) else "retrieval"
            state.latest_query_type = query_type
            shaped = service.context_manager.shape_results(raw, query_type=query_type)
            state.latest_shaped_results = shaped

            state.stage_trace.append(
                {
                    "stage": "execute",
                    "attempt": len(state.previous_queries),
                    "status": "ok",
                    "query_type": query_type,
                    "took_ms": shaped.get("took_ms"),
                    "total_hits": shaped.get("total_hits"),
                }
            )

            return json.dumps(
                {
                    "status": "ok",
                    "query_type": query_type,
                    "took_ms": shaped.get("took_ms"),
                    "total_hits": shaped.get("total_hits"),
                    "results": shaped,
                },
                ensure_ascii=False,
            )

        @tool
        def validate_latest_results() -> str:
            """Validate latest results against the user request."""
            if state.hard_stop:
                return json.dumps(
                    {"status": "hard_stop", "reason": state.hard_stop_reason},
                    ensure_ascii=False,
                )

            if not state.latest_shaped_results:
                state.hard_stop = True
                state.hard_stop_reason = "no_results_to_validate"
                return json.dumps({"status": "hard_stop", "reason": state.hard_stop_reason}, ensure_ascii=False)

            try:
                outcome = asyncio.run(
                    service.validator.validate(
                        question=state.question,
                        shaped_results=state.latest_shaped_results,
                    )
                )
            except Exception as exc:
                logger.exception("validate_latest_results_failed")
                state.hard_stop = True
                state.hard_stop_reason = f"validation_exception: {repr(exc)}"
                return json.dumps({"status": "hard_stop", "reason": state.hard_stop_reason}, ensure_ascii=False)

            state.last_validation = outcome

            if outcome.invalid_names:
                state.observations.append(f"Invalid buckets: {', '.join(outcome.invalid_names[:20])}")

            if outcome.passed:
                state.stage_trace.append(
                    {
                        "stage": "validate",
                        "attempt": len(state.previous_queries),
                        "passed": True,
                        "reason": outcome.reason,
                        "valid_count": outcome.valid_count,
                        "requested_count": outcome.requested_count,
                    }
                )
                return json.dumps(
                    {
                        "status": "validated",
                        "passed": True,
                        "reason": outcome.reason,
                    },
                    ensure_ascii=False,
                )

            state.observations.append(outcome.reason)
            if outcome.refinement_hint:
                state.observations.append(outcome.refinement_hint)

            state.stage_trace.append(
                {
                    "stage": "validate",
                    "attempt": len(state.previous_queries),
                    "passed": False,
                    "reason": outcome.reason,
                    "valid_count": outcome.valid_count,
                    "requested_count": outcome.requested_count,
                    "invalid_names": outcome.invalid_names[:20],
                    "refinement_hint": outcome.refinement_hint,
                }
            )

            if len(state.previous_queries) >= settings.query_max_attempts:
                state.hard_stop = True
                state.hard_stop_reason = "retry_limit_reached_after_validation_failure"
                return json.dumps(
                    {"status": "hard_stop", "reason": state.hard_stop_reason},
                    ensure_ascii=False,
                )

            return json.dumps(
                {
                    "status": "needs_retry",
                    "passed": False,
                    "reason": outcome.reason,
                    "valid_count": outcome.valid_count,
                    "requested_count": outcome.requested_count,
                    "invalid_names": outcome.invalid_names[:20],
                    "refinement_hint": outcome.refinement_hint,
                },
                ensure_ascii=False,
            )

        @tool
        def get_retry_status() -> str:
            """Return retry status."""
            remaining = max(settings.query_max_attempts - len(state.previous_queries), 0)
            return json.dumps(
                {
                    "attempts_used": len(state.previous_queries),
                    "attempts_remaining": remaining,
                    "hard_stop": state.hard_stop,
                    "hard_stop_reason": state.hard_stop_reason,
                    "latest_validation_reason": state.last_validation.reason if state.last_validation else None,
                    "latest_validation_passed": state.last_validation.passed if state.last_validation else None,
                },
                ensure_ascii=False,
            )

        return [build_es_query, execute_latest_query, validate_latest_results, get_retry_status]