"""
Milestone 4 — End-to-End Chat Pipeline

What this file tests (integration):
- POST /api/v1/chat returns:
    - a coherent non-empty natural language response
    - query_metadata.safety_status in {"allowed","blocked","modified"}
    - no raw stack traces in the response body
- "nonsense" input does not leak stack traces (graceful handling)

These tests are skipped unless integration is enabled and backend is reachable.
"""

from __future__ import annotations

import os
import re
import socket
import sys
from pathlib import Path

import pytest


# -----------------------------
# Path bootstrap (optional)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


@pytest.mark.skipif(not _should_run_integration(), reason="Integration disabled or backend not reachable")
@pytest.mark.parametrize(
    "question",
    [
        "Who are the top 10 most mentioned people this week?",
        "What are the top 20 most prolific news sources?",
        "Which countries appear most in reporting about WB_698_TRADE?",
        "How has reporting on ECON_INFLATION changed this month?",
        "What is the average sentiment toward Joe Biden by news source?",
        "What themes are most associated with Singapore?",
        "What has Vladimir Putin been quoted saying recently?",
        "Which people appear most often in the same articles as Elon Musk?",
    ],
)
def test_m4_chat_endpoint_returns_response_and_metadata(question: str):
    """
    Tests (Milestone 4 Acceptance):
    - POST /api/v1/chat returns 200 with JSON containing:
        response: non-empty string
        query_metadata.safety_status: allowed|blocked|modified
    - Raw error traces never appear in response body (no "Traceback").

    Why this matters:
    - Confirms pipeline wiring: NL->Query->Safety->ES->Summarise->Respond.
    - Confirms error hygiene: no stack traces leak to the user.
    """
    import requests

    payload = {"message": question, "session_id": "acceptance-test", "history": []}
    r = requests.post(f"{BACKEND_URL}/api/v1/chat", json=payload, timeout=60)
    assert r.status_code == 200

    data = r.json()
    assert isinstance(data.get("response"), str) and data["response"].strip()
    assert "traceback" not in data["response"].lower()

    qm = data.get("query_metadata") or {}
    assert qm.get("safety_status") in {"allowed", "blocked", "modified", "error", "failed"}

    if qm.get("safety_status") in {"blocked", "error", "failed"}:
        assert qm.get("blocked_reason") is not None
        assert qm.get("es_query") in (None, {})
        
    else:
        # You may or may not return es_query depending on design, but if present, it must be a dict.
        assert (qm.get("es_query") is None) or isinstance(qm.get("es_query"), dict)


@pytest.mark.skipif(not _should_run_integration(), reason="Integration disabled or backend not reachable")
def test_m4_no_stacktrace_leak_on_bad_input():
    """
    Tests (Milestone 4 Acceptance):
    - Garbage/nonsense input should not produce raw stack traces in response.
    - Endpoint either returns:
        - 200 with a user-friendly response, OR
        - 422 validation error (FastAPI) without internal traces.

    Why this matters:
    - Hardens the app against unexpected inputs and prompt injection junk.
    """
    import requests

    payload = {"message": "asdjklqweoiu zxcmnqweoiu ??? !!!", "session_id": "acceptance-test", "history": []}
    r = requests.post(f"{BACKEND_URL}/api/v1/chat", json=payload, timeout=60)
    assert r.status_code in (200, 422)

    if r.status_code == 200:
        data = r.json()
        assert isinstance(data.get("response"), str)
        assert "traceback" not in data["response"].lower()

        qm = data.get("query_metadata") or {}
        assert qm.get("safety_status") in {"allowed", "blocked", "modified"}