# backend/services/query_generator.py

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from config import settings
from services.schema_store import get_schema_store


class QueryGenerationError(Exception):
    """Raised when the LLM output cannot be converted into a valid ES query body."""


class QueryGenerator:
    """Generate Elasticsearch queries from natural language.

    This version no longer hardcodes Appendix A into the prompt.
    Instead, it retrieves the live Elasticsearch mapping, stores it in Chroma,
    and retrieves the most relevant schema chunks for each user question.
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
You are an OSINT assistant that converts a user question into a valid Elasticsearch JSON query body.

Current date and time: {current_time}
Target index: {settings.es_index}

Rules:
1. Return ONLY a JSON object. No prose, no markdown, no code fences.
2. Use only fields that appear in the schema context.
3. Do not invent field names.
4. For "top N", "most common", or ranking questions, use a terms aggregation and set top-level "size": 0.
5. Prefer keyword fields or keyword subfields for exact filters, sorting, and terms aggregations.
6. If a previous attempt found no useful results, broaden the query by simplifying restrictive clauses or widening the time range.
7. Keep the query safe and read-only. Never use scripts.
8. Use concise, production-sensible Elasticsearch queries.

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

    def _parse_json(self, text: str) -> Dict[str, Any]:
        if not text:
            raise QueryGenerationError("Empty LLM output.")

        cleaned = text.strip().replace("```json", "").replace("```", "").strip()
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
            # Safe fallback: query generation can still proceed using the current question,
            # history, and previous attempt observations.
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
        return self._parse_json(str(content))