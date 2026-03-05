"""
Milestone 1 — Infrastructure & Connectivity

What this file tests:
- Backend is reachable and reports healthy dependencies.
- Index stats endpoint returns basic index metrics.

These are integration tests and will be skipped unless:
- RUN_INTEGRATION=1 (or true/yes)
- BACKEND_URL is reachable over TCP
"""

from __future__ import annotations

import os
import re
import socket
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


# -----------------------------
# Config / helpers
# -----------------------------
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _tcp_connectable(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _parse_host_port(url: str) -> tuple[str, int]:
    # expects http://host:port
    m = re.match(r"^https?://([^:/]+)(?::(\d+))?$", url)
    if not m:
        return ("localhost", 8000)
    host = m.group(1)
    port = int(m.group(2) or 80)
    return host, port


def _should_run_integration() -> bool:
    if not RUN_INTEGRATION:
        return False
    host, port = _parse_host_port(BACKEND_URL)
    return _tcp_connectable(host, port)


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.skipif(not _should_run_integration(), reason="Integration disabled or backend not reachable")
def test_m1_health_endpoint_ok_and_indicates_es_llm_reachable():
    """
    Tests (Milestone 1 Acceptance):
    - GET /health returns HTTP 200
    - Response JSON includes:
        status == "ok"
        elasticsearch == True
        llm == True

    Why this matters:
    - Confirms backend is up
    - Confirms it can reach Elasticsearch + LLM (dependency connectivity)
    """
    import requests

    r = requests.get(f"{BACKEND_URL}/health", timeout=10)
    assert r.status_code == 200

    data = r.json()
    assert data.get("status") == "ok"
    assert data.get("elasticsearch") is True
    assert data.get("llm") is True


@pytest.mark.skipif(not _should_run_integration(), reason="Integration disabled or backend not reachable")
def test_m1_index_stats_returns_document_count_and_date_range():
    """
    Tests (Milestone 1 Acceptance):
    - GET /api/v1/index/stats returns:
        total_documents (int >= 0)
        index_size_bytes (int >= 0)
        earliest_date (non-empty str)
        latest_date (non-empty str)
        top_sources (list)

    Why this matters:
    - Confirms backend can query the live ES index for stats used by the UI sidebar.
    """
    import requests

    r = requests.get(f"{BACKEND_URL}/api/v1/index/stats", timeout=20)
    assert r.status_code == 200

    data = r.json()
    assert isinstance(data.get("total_documents"), int)
    assert data["total_documents"] >= 0

    assert isinstance(data.get("index_size_bytes"), int)
    assert data["index_size_bytes"] >= 0

    assert isinstance(data.get("earliest_date"), str) and data["earliest_date"].strip()
    assert isinstance(data.get("latest_date"), str) and data["latest_date"].strip()

    assert isinstance(data.get("top_sources"), list)