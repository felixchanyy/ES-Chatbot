from __future__ import annotations

import asyncio
import copy
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI

from config import settings
from services.schema_store import get_schema_store


class QueryGenerationError(Exception):
    """Raised when the model output cannot be converted into a valid ES query."""


class QueryGenerator:
    """Generate Elasticsearch JSON with schema-aware prompt guidance.

    Retry planner rules:
    - If a previous aggregation field already returned usable buckets, preserve that field on retry.
    - On retry, prefer refining size / excludes / filters instead of changing the aggregation field.
    - Only allow field switching when previous attempts were empty, malformed, or clearly failed structurally.
    """

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )

    def _build_system_prompt(
        self,
        *,
        schema_context: str,
        question: str,
        history_text: str,
        observations_text: str,
        previous_queries_text: str,
        refinement_hint: str,
        retry_plan_text: str,
    ) -> str:
        return f"""
You are an expert Elasticsearch query planner for a GDELT OSINT chatbot.
Return exactly one JSON object and nothing else.

Current index: {settings.es_index}
User question: {question}
Refinement hint: {refinement_hint or '(none)'}

Hard rules:
1. Return valid JSON only. No prose, no markdown, no code fences, no metadata.
2. Use only fields that appear in the schema context, for all dates use V21Date.
3. Do not invent field names. Only use fields that exist in schema context.
4. Never use scripts, updates, deletes, runtime fields, or write operations.
5. For top-N / ranking questions use aggregations and set top-level size to 0.
6. Over-fetch ranking candidates so downstream validation can remove invalid buckets.
7. Prefer keyword fields or keyword subfields for exact filters, sorting, and terms aggregations.
8. Preserve requested count, entity type, timeframe, country/region/category constraints, and co-occurrence intent.
9. If earlier attempts had invalid entities or too few valid entities, refine the query instead of returning the same query again.
10. If an earlier aggregation field already returned usable results, preserve that same field on retry and refine around it.
11. Only switch aggregation field if earlier attempts were empty, malformed, or clearly failed structurally.
12. The query must be concise, production-sensible, read-only Elasticsearch.
13. If no specific time range is provided, default to a 1 year timeframe.


Recent chat history:
{history_text}

Previous generated queries:
{previous_queries_text}

Observations from validation / execution:
{observations_text}

Retry planner guidance:
{retry_plan_text}

Schema context:
{schema_context}
""".strip()

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise QueryGenerationError("Empty LLM output.")

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        candidate = fenced.group(1) if fenced else text

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if not match:
                raise QueryGenerationError("LLM did not return valid JSON.")
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise QueryGenerationError(f"Invalid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise QueryGenerationError("LLM output was JSON but not an object.")
        return parsed

    @staticmethod
    def _format_history(history: Optional[List[Dict[str, Any]]]) -> str:
        if not history:
            return "(none)"
        return "\n".join(
            f"{item.get('role', 'user').upper()}: {item.get('content', '')}" for item in history[-8:]
        )

    @staticmethod
    def _normalize_aggs_key(query: Dict[str, Any]) -> Dict[str, Any]:
        if "aggregations" in query and "aggs" not in query:
            query["aggs"] = query.pop("aggregations")
        return query

    @staticmethod
    def _is_ranking_question(question: str) -> bool:
        q = question.lower()
        return any(term in q for term in ["top ", "most mentioned", "most common", "rank", "ranking"])

    @staticmethod
    def _extract_requested_top_n(question: str) -> Optional[int]:
        match = re.search(r"\btop\s+(\d+)\b", question, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    @staticmethod
    def _iter_terms_aggs(query: Dict[str, Any]) -> List[Dict[str, Any]]:
        aggs = query.get("aggs")
        if not isinstance(aggs, dict):
            return []
        found: List[Dict[str, Any]] = []

        def walk(node: Dict[str, Any]) -> None:
            for _, value in node.items():
                if not isinstance(value, dict):
                    continue
                if isinstance(value.get("terms"), dict):
                    found.append(value["terms"])
                nested = value.get("aggs") or value.get("aggregations")
                if isinstance(nested, dict):
                    walk(nested)

        walk(aggs)
        return found

    @staticmethod
    def _extract_invalid_bucket_names(observations: List[str]) -> List[str]:
        invalid_names: List[str] = []
        patterns = [
            r"Invalid buckets:\s*(.+)",
            r"Rejected buckets:\s*(.+)",
            r"invalid entities:\s*(.+)",
        ]
        for obs in observations[-8:]:
            for pattern in patterns:
                match = re.search(pattern, obs, flags=re.IGNORECASE)
                if not match:
                    continue
                parts = [p.strip(" .'\"") for p in re.split(r",|\||;", match.group(1)) if p.strip()]
                invalid_names.extend(parts)

        deduped: List[str] = []
        seen = set()
        for name in invalid_names:
            if name not in seen:
                deduped.append(name)
                seen.add(name)
        return deduped

    @staticmethod
    def _query_has_v21date_range(query: Dict[str, Any]) -> bool:
        def walk(node: Any) -> bool:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "range" and isinstance(value, dict) and "V21Date" in value:
                        return True
                    if walk(value):
                        return True
            elif isinstance(node, list):
                return any(walk(item) for item in node)
            return False

        return walk(query)

    @staticmethod
    def _extract_terms_field(query: Dict[str, Any]) -> Optional[str]:
        """Return the first terms aggregation field found."""
        aggs = query.get("aggs")
        if not isinstance(aggs, dict):
            return None

        result: Optional[str] = None

        def walk(node: Dict[str, Any]) -> None:
            nonlocal result
            if result is not None:
                return
            for _, value in node.items():
                if not isinstance(value, dict):
                    continue
                terms = value.get("terms")
                if isinstance(terms, dict) and isinstance(terms.get("field"), str):
                    result = terms["field"]
                    return
                nested = value.get("aggs") or value.get("aggregations")
                if isinstance(nested, dict):
                    walk(nested)

        walk(aggs)
        return result

    @staticmethod
    def _set_terms_field(query: Dict[str, Any], forced_field: str) -> Dict[str, Any]:
        """Replace all terms agg fields with forced_field."""
        query = copy.deepcopy(query)
        aggs = query.get("aggs")
        if not isinstance(aggs, dict):
            return query

        def walk(node: Dict[str, Any]) -> None:
            for _, value in node.items():
                if not isinstance(value, dict):
                    continue
                terms = value.get("terms")
                if isinstance(terms, dict):
                    terms["field"] = forced_field
                nested = value.get("aggs") or value.get("aggregations")
                if isinstance(nested, dict):
                    walk(nested)

        walk(aggs)
        return query

    @staticmethod
    def _extract_latest_matching_observation(
        observations: List[str], patterns: List[str]
    ) -> Optional[str]:
        for obs in reversed(observations[-12:]):
            for pattern in patterns:
                if re.search(pattern, obs, flags=re.IGNORECASE):
                    return obs
        return None

    def _apply_default_time_range(self, question: str, query: Dict[str, Any]) -> Dict[str, Any]:
        explicit_time_terms = [
            "today",
            "yesterday",
            "this week",
            "last week",
            "this month",
            "last month",
            "this year",
            "last year",
            "past ",
            "ago",
            "between ",
            "from ",
            "since ",
            "before ",
            "after ",
            "in 20",
            "last 5 weeks",
            "last 7 days",
            "past 7 days",
        ]
        if any(term in question.lower() for term in explicit_time_terms):
            return query
        if self._query_has_v21date_range(query):
            return query

        query = copy.deepcopy(query)
        one_year_ago = (datetime.utcnow() - timedelta(days=365)).strftime("%Y%m%d")
        bool_query = query.setdefault("query", {}).setdefault("bool", {})
        filters = bool_query.setdefault("filter", [])
        if isinstance(filters, list):
            filters.append({"range": {"V21Date": {"gte": one_year_ago, "format": "yyyyMMdd"}}})
        return query

    def _repair_range_queries(self, node: Any) -> Any:
        """
        Fix common LLM mistakes like:
        {"range": {"V21Date": {"gte": "20260101"}, "format": "yyyyMMdd"}}
        into:
        {"range": {"V21Date": {"gte": "20260101", "format": "yyyyMMdd"}}}
        """
        if isinstance(node, list):
            return [self._repair_range_queries(item) for item in node]

        if not isinstance(node, dict):
            return node

        repaired = {}
        for key, value in node.items():
            if key == "range" and isinstance(value, dict):
                range_fields = []
                extra_keys = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        range_fields.append((k, v))
                    else:
                        extra_keys[k] = v

                if len(range_fields) == 1:
                    field_name, field_body = range_fields[0]
                    field_body = dict(field_body)
                    for ek, ev in extra_keys.items():
                        if ek in {"format", "time_zone", "boost", "relation"}:
                            field_body[ek] = ev
                    repaired[key] = {field_name: self._repair_range_queries(field_body)}
                else:
                    repaired[key] = self._repair_range_queries(value)
            else:
                repaired[key] = self._repair_range_queries(value)

        return repaired

    def _extract_retry_plan(
        self,
        *,
        question: str,
        observations: List[str],
        previous_queries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build an enforced retry plan from prior attempts.

        Core policy:
        - Preserve last working terms field if there was evidence it returned data.
        - Only switch field when prior attempts were empty / malformed / execution-failed.
        """
        plan: Dict[str, Any] = {
            "has_previous_queries": bool(previous_queries),
            "preserve_terms_field": None,
            "allow_field_switch": True,
            "reason": "no_previous_queries",
            "retry_mode": "fresh",
        }

        if not self._is_ranking_question(question):
            plan["reason"] = "non_ranking_question"
            plan["retry_mode"] = "general"
            return plan

        if not previous_queries:
            plan["reason"] = "first_attempt"
            plan["retry_mode"] = "fresh"
            return plan

        last_query = self._normalize_aggs_key(copy.deepcopy(previous_queries[-1]))
        last_field = self._extract_terms_field(last_query)

        got_results_obs = self._extract_latest_matching_observation(
            observations,
            [
                r"got results",
                r"returned results",
                r"with results",
                r"total_hits[\"'=:\s]+[1-9]\d*",
                r"enough_valid",
                r"only_\d+_valid_items_for_top_\d+",
                r"invalid buckets:",
            ],
        )
        empty_obs = self._extract_latest_matching_observation(
            observations,
            [
                r"empty results",
                r"no results",
                r"no data",
                r"aggregation returned empty",
                r"top_people\":\s*\[\]",
                r"top_people'\s*:\s*\[\]",
            ],
        )
        execution_failed_obs = self._extract_latest_matching_observation(
            observations,
            [
                r"execution_failed",
                r"malformed query",
                r"parsing_exception",
                r"badrequesterror",
                r"query_generation_error",
            ],
        )

        # Strongest rule:
        # if there was evidence that a prior ranking field produced usable results,
        # preserve the latest working field.
        if got_results_obs and last_field:
            plan["preserve_terms_field"] = last_field
            plan["allow_field_switch"] = False
            plan["reason"] = "previous_field_worked_preserve_it"
            plan["retry_mode"] = "refine_same_field"
            return plan

        # If the latest attempt was empty or malformed, it is okay to switch field.
        if (empty_obs or execution_failed_obs) and last_field:
            # Try to find an earlier field that worked.
            for prev in reversed(previous_queries[:-1]):
                prev_norm = self._normalize_aggs_key(copy.deepcopy(prev))
                prev_field = self._extract_terms_field(prev_norm)
                if prev_field:
                    plan["preserve_terms_field"] = prev_field
                    plan["allow_field_switch"] = False
                    plan["reason"] = "latest_attempt_failed_revert_to_last_known_field"
                    plan["retry_mode"] = "revert_and_refine"
                    return plan

            plan["preserve_terms_field"] = None
            plan["allow_field_switch"] = True
            plan["reason"] = "latest_attempt_failed_no_known_working_field"
            plan["retry_mode"] = "field_exploration_allowed"
            return plan

        # Default retry policy on ranking questions with prior attempts:
        # keep the latest field unless explicit evidence says it failed structurally.
        if last_field:
            plan["preserve_terms_field"] = last_field
            plan["allow_field_switch"] = False
            plan["reason"] = "default_preserve_latest_field"
            plan["retry_mode"] = "refine_same_field"
            return plan

        plan["reason"] = "no_terms_field_detected"
        plan["retry_mode"] = "general"
        return plan

    def _format_retry_plan_text(self, plan: Dict[str, Any]) -> str:
        return json.dumps(plan, ensure_ascii=False, indent=2)

    def _enforce_retry_plan(
        self,
        *,
        query: Dict[str, Any],
        retry_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply deterministic retry planner constraints after LLM generation."""
        query = copy.deepcopy(query)
        forced_field = retry_plan.get("preserve_terms_field")
        allow_switch = bool(retry_plan.get("allow_field_switch", True))

        if forced_field and not allow_switch:
            query = self._set_terms_field(query, forced_field)

        return query

    def _postprocess(
        self,
        *,
        question: str,
        query: Dict[str, Any],
        observations: List[str],
        retry_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        query = self._normalize_aggs_key(copy.deepcopy(query))
        query = self._repair_range_queries(query)
        query = self._enforce_retry_plan(query=query, retry_plan=retry_plan)

        if self._is_ranking_question(question) and "aggs" in query:
            query["size"] = 0
            requested_n = self._extract_requested_top_n(question)
            desired_size = max((requested_n or 10) * 3, (requested_n or 10) + 10)

            if any("only_" in obs.lower() or "insufficient" in obs.lower() for obs in observations[-6:]):
                desired_size = max(desired_size, 50)

            invalid_names = self._extract_invalid_bucket_names(observations)
            for terms_def in self._iter_terms_aggs(query):
                current_size = terms_def.get("size")
                if not isinstance(current_size, int) or current_size < desired_size:
                    terms_def["size"] = desired_size
                if invalid_names and terms_def.get("exclude") is None:
                    terms_def["exclude"] = invalid_names

        return self._apply_default_time_range(question, query)

    async def build_query(
        self,
        *,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
        observations: Optional[List[str]] = None,
        previous_queries: Optional[List[Dict[str, Any]]] = None,
        refinement_hint: str = "",
    ) -> Dict[str, Any]:
        observations = observations or []
        previous_queries = previous_queries or []

        schema_context = "(schema retrieval unavailable)"
        try:
            schema_store = get_schema_store()
            schema_docs = await schema_store.search_schema(question, k=settings.schema_search_k)
            if not schema_docs:
                schema_docs = await schema_store.get_schema_overview(limit=12)
            if schema_docs:
                schema_context = "\n".join(schema_docs)
        except Exception:
            pass

        retry_plan = self._extract_retry_plan(
            question=question,
            observations=observations,
            previous_queries=previous_queries,
        )

        system_prompt = self._build_system_prompt(
            schema_context=schema_context,
            question=question,
            history_text=self._format_history(history),
            observations_text="\n".join(observations) or "(none)",
            previous_queries_text=json.dumps(previous_queries, ensure_ascii=False),
            refinement_hint=refinement_hint,
            retry_plan_text=self._format_retry_plan_text(retry_plan),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        response = await asyncio.to_thread(self.llm.invoke, messages)
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)

        parsed = self._extract_json(str(content))
        return self._postprocess(
            question=question,
            query=parsed,
            observations=observations,
            retry_plan=retry_plan,
        )