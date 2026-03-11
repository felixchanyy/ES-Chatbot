"""
Milestone 2 acceptance tests for the newer async, schema-aware QueryGenerator.

This version does NOT assume QueryGenerator has a private helper like
`_get_schema_context_async`. Instead, it patches the dependency actually used
by the module: `get_schema_store()` inside services.query_generator.

These tests validate:
1. Natural language -> valid Elasticsearch query dict
2. Correct use of V2Persons.V1Person.keyword for top-people aggregation
3. Invalid LLM JSON raises QueryGenerationError
4. Retrieved schema context is included in the LLM system prompt
5. Empty schema context does not crash query generation
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


class _FakeLLM:
    """Fake LLM returning a fixed `.content` string."""

    def __init__(self, content: str):
        self._content = content
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return SimpleNamespace(content=self._content)


class _FakeSchemaStore:
    """Fake schema store matching the newer async schema retrieval contract."""

    def __init__(self, schema_docs=None, overview_docs=None):
        self.schema_docs = schema_docs or []
        self.overview_docs = overview_docs or []

    async def search_schema(self, question: str, k: int = 8):
        return self.schema_docs

    async def get_schema_overview(self, limit: int = 12):
        return self.overview_docs


@pytest.mark.asyncio
async def test_m2_query_generation_returns_valid_es_query_dict_for_top10_people_this_week(monkeypatch):
    """
    Acceptance criteria:
    - Asking "Who are the top 10 people mentioned this week?"
      returns a valid ES query JSON dict
    - Query uses V2Persons.V1Person.keyword in a terms aggregation
    - Includes a time range filter
    """
    from services.query_generator import QueryGenerator

    llm_json = json.dumps(
        {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [
                        {
                            "range": {
                                "V21Date": {
                                    "gte": "now-7d/d",
                                    "lte": "now",
                                }
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "top_people": {
                    "terms": {
                        "field": "V2Persons.V1Person.keyword",
                        "size": 10,
                    }
                }
            },
        }
    )

    qg = QueryGenerator()
    monkeypatch.setattr(qg, "llm", _FakeLLM(llm_json), raising=True)

    fake_schema_store = _FakeSchemaStore(
        schema_docs=[
            "Field: V21Date. Type: date. Usage: Use for date filtering.",
            "Field: V2Persons.V1Person.keyword. Type: keyword. Usage: Use for exact matches and terms aggregations.",
        ]
    )

    monkeypatch.setattr(
        "services.query_generator.get_schema_store",
        lambda: fake_schema_store,
    )

    result = await qg.generate("Who are the top 10 people mentioned this week?")

    assert isinstance(result, dict)
    assert "aggs" in result
    assert "top_people" in result["aggs"]

    terms = result["aggs"]["top_people"]["terms"]
    assert terms["field"] == "V2Persons.V1Person.keyword"
    assert terms["size"] == 10

    assert "query" in result
    filters = result["query"]["bool"]["filter"]
    assert any("range" in item and "V21Date" in item["range"] for item in filters)


@pytest.mark.asyncio
async def test_m2_invalid_llm_output_raises_query_generation_error(monkeypatch):
    """
    Acceptance criteria:
    - If the LLM returns invalid JSON, QueryGenerator raises QueryGenerationError
    """
    from services.query_generator import QueryGenerationError, QueryGenerator

    qg = QueryGenerator()
    monkeypatch.setattr(qg, "llm", _FakeLLM("```json\n{not valid}\n```"), raising=True)

    fake_schema_store = _FakeSchemaStore(
        schema_docs=[
            "Field: V21Date. Type: date.",
            "Field: V2Persons.V1Person.keyword. Type: keyword.",
        ]
    )

    monkeypatch.setattr(
        "services.query_generator.get_schema_store",
        lambda: fake_schema_store,
    )

    with pytest.raises(QueryGenerationError) as exc_info:
        await qg.generate("Who are the top 10 people mentioned this week?")

    assert "Invalid JSON" in str(exc_info.value) or "valid JSON" in str(exc_info.value)


@pytest.mark.asyncio
async def test_m2_schema_context_is_included_in_prompt(monkeypatch):
    """
    Regression test for the schema-aware design:
    verifies that retrieved schema context is inserted into the LLM system prompt.
    """
    from services.query_generator import QueryGenerator

    llm_json = json.dumps(
        {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [
                        {
                            "range": {
                                "V21Date": {
                                    "gte": "now-7d/d",
                                    "lte": "now",
                                }
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "top_people": {
                    "terms": {
                        "field": "V2Persons.V1Person.keyword",
                        "size": 10,
                    }
                }
            },
        }
    )

    fake_llm = _FakeLLM(llm_json)
    qg = QueryGenerator()
    monkeypatch.setattr(qg, "llm", fake_llm, raising=True)

    schema_context_docs = [
        "Field: V2Persons.V1Person.keyword. Type: keyword. Usage: Use for exact matches and terms aggregations."
    ]
    fake_schema_store = _FakeSchemaStore(schema_docs=schema_context_docs)

    monkeypatch.setattr(
        "services.query_generator.get_schema_store",
        lambda: fake_schema_store,
    )

    await qg.generate("Who are the top 10 people mentioned this week?")

    sent_messages = fake_llm.last_messages
    assert isinstance(sent_messages, list)
    assert len(sent_messages) >= 2

    system_message = sent_messages[0]
    assert system_message["role"] == "system"
    assert schema_context_docs[0] in system_message["content"]


@pytest.mark.asyncio
async def test_m2_generate_can_use_empty_schema_context_without_crashing(monkeypatch):
    """
    If schema search returns nothing, QueryGenerator should fall back to overview
    or continue without crashing.
    """
    from services.query_generator import QueryGenerator

    llm_json = json.dumps(
        {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [
                        {
                            "range": {
                                "V21Date": {
                                    "gte": "now-7d/d",
                                    "lte": "now",
                                }
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "top_people": {
                    "terms": {
                        "field": "V2Persons.V1Person.keyword",
                        "size": 10,
                    }
                }
            },
        }
    )

    qg = QueryGenerator()
    monkeypatch.setattr(qg, "llm", _FakeLLM(llm_json), raising=True)

    fake_schema_store = _FakeSchemaStore(
        schema_docs=[],
        overview_docs=["(no schema context available)"],
    )

    monkeypatch.setattr(
        "services.query_generator.get_schema_store",
        lambda: fake_schema_store,
    )

    result = await qg.generate("Who are the top 10 people mentioned this week?")
    assert isinstance(result, dict)
    assert result["aggs"]["top_people"]["terms"]["field"] == "V2Persons.V1Person.keyword"