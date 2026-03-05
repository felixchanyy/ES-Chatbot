"""
Milestone 3 — Query Safety Layer

What this file tests (unit-level, offline):
- Blocks "script" key at any depth (script injection defence)
- Caps top-level `size` to MAX_RESULT_DOCS
- Injects `_source.excludes` ALWAYS_EXCLUDE_FIELDS
- Caps terms aggregation bucket sizes to MAX_AGG_BUCKETS
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def test_m3_blocks_script_key_at_any_depth():
    """
    Tests (Milestone 3 Acceptance):
    - Any query containing a key named "script" anywhere in the nested structure
      is BLOCKED and returns:
        status == SafetyStatus.BLOCKED
        query == None
        reason == "script_detected"

    Why this matters:
    - Prevents Elasticsearch script execution via LLM-generated payloads.
    """
    from services.query_safety import QuerySafetyLayer, SafetyStatus

    safety = QuerySafetyLayer(max_result_docs=20, max_agg_buckets=50)

    res = safety.validate({"query": {"bool": {"filter": [{"script": {"source": "1+1"}}]}}})
    assert res.status == SafetyStatus.BLOCKED
    assert res.query is None
    assert res.reason == "script_detected"


def test_m3_caps_top_level_size_to_max_result_docs():
    """
    Tests (Milestone 3 Acceptance):
    - If query sets `size` larger than max_result_docs (e.g. 10000),
      the safety layer MODIFIES the query so size == max_result_docs.

    Why this matters:
    - Prevents runaway result retrieval / token blowups / resource exhaustion.
    """
    from services.query_safety import QuerySafetyLayer, SafetyStatus

    safety = QuerySafetyLayer(max_result_docs=20, max_agg_buckets=50)

    res = safety.validate({"query": {"match_all": {}}, "size": 10000})
    assert res.status in (SafetyStatus.MODIFIED, SafetyStatus.ALLOWED)
    assert res.query is not None
    assert res.query.get("size") == 20


def test_m3_injects_source_excludes_always_exclude_fields():
    """
    Tests (Milestone 3 Acceptance):
    - `_source.excludes` must be injected on every query, even if absent originally.
    - It must include every field in ALWAYS_EXCLUDE_FIELDS.

    Why this matters:
    - Prevents huge fields (e.g. event.original) from reaching the LLM/context manager.
    """
    from services.query_safety import QuerySafetyLayer, ALWAYS_EXCLUDE_FIELDS

    safety = QuerySafetyLayer(max_result_docs=20, max_agg_buckets=50)

    res = safety.validate({"query": {"match_all": {}}, "size": 1})
    assert res.query is not None
    assert "_source" in res.query
    assert "excludes" in res.query["_source"]

    excludes = res.query["_source"]["excludes"]
    for f in ALWAYS_EXCLUDE_FIELDS:
        assert f in excludes


def test_m3_caps_terms_aggregation_bucket_size():
    """
    Tests (Milestone 3 Acceptance):
    - If any `terms` aggregation requests size > max_agg_buckets,
      it must be capped down to max_agg_buckets.

    Why this matters:
    - Prevents large bucket explosions and slow aggregations.
    """
    from services.query_safety import QuerySafetyLayer

    safety = QuerySafetyLayer(max_result_docs=20, max_agg_buckets=50)

    res = safety.validate(
        {
            "size": 0,
            "aggs": {
                "top_people": {"terms": {"field": "V2Persons.V1Person.keyword", "size": 1000}}
            },
        }
    )
    assert res.query is not None
    assert res.query["aggs"]["top_people"]["terms"]["size"] == 50