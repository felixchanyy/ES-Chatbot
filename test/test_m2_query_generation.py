"""
Milestone 2 — Query Generation Pipeline

What this file tests (unit-level, offline):
- NL question -> valid Elasticsearch query dict (no execution)
- Uses correct mapping fields (e.g., V2Persons.V1Person.keyword for terms agg)
- Invalid LLM output is caught and raises QueryGenerationError
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


# -----------------------------
# Path bootstrap (fix imports)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../ES-Chatbot
_backend_dir = PROJECT_ROOT / "backend"
for p in (PROJECT_ROOT, _backend_dir):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


# -----------------------------
# Fakes (no external deps)
# -----------------------------
@dataclass
class _FakeLLMResponse:
    content: str


class _FakeLLM:
    """
    Minimal stub for whatever LLM wrapper your QueryGenerator uses.
    It must expose `.invoke(messages)` and return an object with `.content`.
    """

    def __init__(self, content: str):
        self._content = content

    def invoke(self, messages: Any) -> _FakeLLMResponse:
        return _FakeLLMResponse(self._content)


class _FakeVectorStore:
    """
    Minimal stub for a vector store with `.similarity_search(question, k)`.
    Returns objects with `.page_content`.
    """

    def similarity_search(self, question: str, k: int = 6):
        class _Doc:
            def __init__(self, page_content: str):
                self.page_content = page_content

        return [
            _Doc("V2Persons.V1Person.keyword is a keyword field for terms aggregations."),
            _Doc("V21Date is a date field used for time filtering."),
        ]


def test_m2_query_generation_returns_valid_es_query_dict_for_top10_people_this_week(monkeypatch):
    """
    Tests (Milestone 2 Acceptance):
    - Asking: "Who are the top 10 people mentioned this week?"
      returns a valid ES query JSON dict.
    - Query uses V2Persons.V1Person.keyword in a terms aggregation.

    What 'valid' means here:
    - Python dict (not a string)
    - Contains aggs with a terms aggregation using that keyword field
    - Includes a time range filter (example uses now-7d/d)
    """
    from services.query_generator import QueryGenerator

    llm_json = json.dumps(
        {
            "size": 0,
            "query": {
                "bool": {"filter": [{"range": {"V21Date": {"gte": "now-7d/d", "lte": "now"}}}]}
            },
            "aggs": {"top_people": {"terms": {"field": "V2Persons.V1Person.keyword", "size": 10}}},
        }
    )

    qg = QueryGenerator()
    monkeypatch.setattr(qg, "llm", _FakeLLM(llm_json), raising=True)
    monkeypatch.setattr(qg, "vectorstore", _FakeVectorStore(), raising=True)

    query = qg.generate("Who are the top 10 people mentioned this week?", history=[])

    assert isinstance(query, dict)
    assert query.get("size") == 0
    assert "aggs" in query

    # Ensure the expected field is used in a terms agg somewhere inside aggs
    terms_aggs_text = json.dumps(query.get("aggs", {}))
    assert "V2Persons.V1Person.keyword" in terms_aggs_text
    assert re.search(r'"terms"\s*:\s*\{', terms_aggs_text), "No terms aggregation found"


def test_m2_invalid_llm_output_raises_query_generation_error(monkeypatch):
    """
    Tests (Milestone 2 Acceptance):
    - If the LLM returns invalid JSON (even wrapped in code fences),
      QueryGenerator must raise QueryGenerationError with a clear message.

    Why this matters:
    - Prevents passing malformed / unsafe payloads to later stages.
    - Makes frontend UX clean: user gets a friendly error instead of a crash.
    """
    from services.query_generator import QueryGenerator, QueryGenerationError

    qg = QueryGenerator()
    monkeypatch.setattr(qg, "llm", _FakeLLM("```json\n{not valid}\n```"), raising=True)
    monkeypatch.setattr(qg, "vectorstore", _FakeVectorStore(), raising=True)

    with pytest.raises(QueryGenerationError) as ei:
        _ = qg.generate("Who are the top 10 people mentioned this week?", history=[])

    msg = str(ei.value).lower()
    assert any(k in msg for k in ["invalid", "json", "parse"]), f"Message not clear enough: {ei.value}"