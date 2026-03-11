# backend/services/query_generator.py

from __future__ import annotations

import asyncio
import copy
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from config import settings
from services.schema_store import get_schema_store


class QueryGenerationError(Exception):
    """Raised when the LLM output cannot be converted into a valid Elasticsearch query body."""


class QueryGenerator:
    """Generate Elasticsearch queries from natural language.

    This version:
    - retrieves live schema context from Chroma
    - supports prior attempts / observations
    - strengthens top-N ranking query generation
    - applies deterministic post-processing to reduce shallow failure modes
    """

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            openai_api_base=settings.llm_base_url,
            openai_api_key=settings.llm_api_key,
            model_name=settings.llm_model_name,
            temperature=0,
        )

    def _build_system_prompt(
        self,
        *,
        schema_context: str,
        current_time: str,
        history_text: str,
        prior_attempts_text: str,
        observation_text: str,
    ) -> str:
        return f"""
You are an OSINT assistant that converts a user question into a valid Elasticsearch JSON query.

Current date and time: {current_time}
Target index: {settings.es_index}

Rules:
1. Return ONLY one valid JSON object. No prose, no markdown, no code fences, no explanations.
2. Use only fields that appear in the schema context. For all date filtering, use V21Date unless the schema clearly requires otherwise.
3. Do not invent field names.
4. Keep the query read-only and safe. Never use scripts, updates, deletes, runtime mappings, stored scripts, or inline script logic.
5. Prefer keyword fields or keyword subfields for exact filters, sorting, and terms aggregations.
6. For "top N", "most common", "most mentioned", "rank", or similar ranking questions:
   - use a terms aggregation
   - set top-level "size": 0
   - return enough extra candidate buckets so downstream filtering still has enough valid items
   - therefore do NOT set the aggregation size equal to N unless explicitly required
7. If a previous attempt returned insufficient valid ranked items, broaden the candidate pool by increasing aggregation size and excluding already-known bad buckets when appropriate.
8. If a previous attempt found no useful results, broaden the query by simplifying restrictive clauses or widening the time range.
9. If no specific time range is provided, default to a 1 year timeframe.
10. Use concise, production-sensible Elasticsearch queries.
11. Preserve the user intent exactly:
   - requested entity type
   - requested count
   - region / country filters
   - timeframe
12. For aggregation questions, do not return document retrieval unless clearly needed.

Conversation history:
{history_text}

Previous generated queries:
{prior_attempts_text}

Observations from previous attempts:
{observation_text}

Relevant live schema context retrieved from Chroma:
{schema_context}
""".strip()

    def _format_history(self, history: Optional[List[Dict[str, Any]]]) -> str:
        if not history:
            return "(none)"
        lines: List[str] = []
        for item in history[-6:]:
            role = item.get("role", "user")
            content = item.get("content", "")
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def _extract_query_block(self, text: str) -> str | None:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)

        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1)

        return None

    def _parse_json(self, text: str) -> Dict[str, Any]:
        if not text:
            raise QueryGenerationError("Empty LLM output.")

        text = text.strip()
        block = self._extract_query_block(text)
        cleaned = block if block else text

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise QueryGenerationError("LLM output was JSON but not an object.")
            return parsed
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise QueryGenerationError("LLM did not return valid JSON.")
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise QueryGenerationError(f"Invalid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise QueryGenerationError("LLM output was JSON but not an object.")
            return parsed

    def _extract_requested_top_n(self, question: str) -> Optional[int]:
        match = re.search(r"\btop\s+(\d+)\b", question, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

        match = re.search(r"\bmost\s+mentioned\s+(\d+)\b", question, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None

    def _is_ranking_question(self, question: str) -> bool:
        q = question.lower()
        ranking_terms = [
            "top ",
            "most mentioned",
            "most common",
            "most frequent",
            "highest",
            "rank",
            "ranking",
        ]
        return any(term in q for term in ranking_terms)

    def _normalize_aggs_key(self, query: Dict[str, Any]) -> Dict[str, Any]:
        if "aggregations" in query and "aggs" not in query:
            query["aggs"] = query.pop("aggregations")
        return query

    def _iter_terms_aggs(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                nested_aggs = value.get("aggs") or value.get("aggregations")
                if isinstance(nested_aggs, dict):
                    walk(nested_aggs)

        walk(aggs)
        return found

    def _extract_invalid_bucket_names(self, observations: Optional[List[str]]) -> List[str]:
        if not observations:
            return []

        invalid_names: List[str] = []

        patterns = [
            r"Invalid buckets:\s*(.+)",
            r"Rejected buckets:\s*(.+)",
            r"bad buckets:\s*(.+)",
            r"invalid entities:\s*(.+)",
        ]

        for obs in observations[-6:]:
            for pattern in patterns:
                match = re.search(pattern, obs, flags=re.IGNORECASE)
                if not match:
                    continue
                raw = match.group(1).strip()
                parts = [p.strip(" .'\"") for p in re.split(r",|\||;", raw) if p.strip()]
                invalid_names.extend(parts)

        # de-dup while preserving order
        seen = set()
        deduped: List[str] = []
        for name in invalid_names:
            if name not in seen:
                deduped.append(name)
                seen.add(name)
        return deduped

    def _needs_more_candidates(self, observations: Optional[List[str]]) -> bool:
        if not observations:
            return False

        joined = "\n".join(observations[-6:]).lower()
        triggers = [
            "insufficient",
            "only_",
            "need more valid entities",
            "fewer than requested",
            "not enough valid",
            "returned only",
            "top 10",
            "top n",
        ]
        return any(t in joined for t in triggers)

    def _apply_time_default(self, question: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """If no explicit timeframe is mentioned and query lacks a V21Date range, add a 1-year default."""
        q = question.lower()

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
            "last ",
            "ago",
            "between ",
            "from ",
            "since ",
            "before ",
            "after ",
            "in 20",
        ]
        if any(term in q for term in explicit_time_terms):
            return query

        if self._query_has_v21date_range(query):
            return query

        query = copy.deepcopy(query)
        one_year_ago = (datetime.utcnow() - timedelta(days=365)).strftime("%Y%m%d")
        bool_query = query.setdefault("query", {}).setdefault("bool", {})
        filters = bool_query.setdefault("filter", [])
        if isinstance(filters, list):
            filters.append({"range": {"V21Date": {"gte": one_year_ago}}})
        return query

    def _query_has_v21date_range(self, query: Dict[str, Any]) -> bool:
        def walk(node: Any) -> bool:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "range" and isinstance(value, dict) and "V21Date" in value:
                        return True
                    if walk(value):
                        return True
            elif isinstance(node, list):
                for item in node:
                    if walk(item):
                        return True
            return False

        return walk(query)

    def _postprocess_query(
        self,
        *,
        question: str,
        query: Dict[str, Any],
        observations: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Deterministic cleanup and strengthening of generated query."""
        query = copy.deepcopy(query)
        query = self._normalize_aggs_key(query)

        # Ranking queries should be aggregations with top-level size 0.
        is_ranking = self._is_ranking_question(question)
        requested_n = self._extract_requested_top_n(question)

        if is_ranking and "aggs" in query:
            query["size"] = 0

            terms_aggs = self._iter_terms_aggs(query)
            if terms_aggs:
                # Over-fetch so post-filtering does not collapse top-N too early.
                if requested_n is not None:
                    desired_size = max(requested_n * 3, requested_n + 10)
                else:
                    desired_size = 25

                if self._needs_more_candidates(observations):
                    desired_size = max(desired_size, 50)

                for terms_def in terms_aggs:
                    current_size = terms_def.get("size")
                    if not isinstance(current_size, int) or current_size < desired_size:
                        terms_def["size"] = desired_size

                # If observations told us which buckets were invalid, exclude them.
                invalid_names = self._extract_invalid_bucket_names(observations)
                if invalid_names:
                    for terms_def in terms_aggs:
                        existing_exclude = terms_def.get("exclude")
                        if existing_exclude is None:
                            terms_def["exclude"] = invalid_names
                        elif isinstance(existing_exclude, list):
                            merged = list(dict.fromkeys(existing_exclude + invalid_names))
                            terms_def["exclude"] = merged
                        elif isinstance(existing_exclude, str):
                            # Preserve existing regex exclude; do not overwrite.
                            pass

        query = self._apply_time_default(question, query)
        return query

    async def generate(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
        previous_queries: Optional[List[Dict[str, Any]]] = None,
        observations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate an Elasticsearch query body using live schema retrieval."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        schema_context = "(no schema context available)"
        try:
            schema_store = get_schema_store()
            schema_docs = await schema_store.search_schema(question, k=8)
            if not schema_docs:
                schema_docs = await schema_store.get_schema_overview(limit=12)
            if schema_docs:
                schema_context = "\n".join(schema_docs)
        except Exception:
            schema_context = "(schema retrieval unavailable)"

        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(
                    schema_context=schema_context,
                    current_time=now,
                    history_text=self._format_history(history),
                    prior_attempts_text=json.dumps(previous_queries or [], ensure_ascii=False),
                    observation_text="\n".join(observations or []) or "(none)",
                ),
            },
            {"role": "user", "content": question},
        ]

        response = await asyncio.to_thread(self.llm.invoke, messages)
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)

        parsed = self._parse_json(str(content))
        parsed = self._postprocess_query(
            question=question,
            query=parsed,
            observations=observations,
        )
        return parsed