"""
Milestone 5 — Streamlit Frontend

What this file tests (offline / static checks):
- Chat history persistence via st.session_state
- Sidebar loads index stats at startup (heuristic)
- "Show raw query" toggle exists somewhere in frontend code
- Frontend does NOT obviously print tracebacks to UI

These do not require running Streamlit; they scan source files.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../ES-Chatbot
_frontend_dir = PROJECT_ROOT / "frontend"

# (Optional) add to path for imports if you ever want
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_m5_frontend_uses_session_state_for_chat_history():
    """
    Tests (Milestone 5 Acceptance):
    - Chat history persists across interactions within a session.

    Implementation expectation:
    - frontend/app.py uses st.session_state (e.g. st.session_state["messages"]).
    """
    app_py = _frontend_dir / "app.py"
    assert app_py.exists(), "frontend/app.py missing"

    text = app_py.read_text(encoding="utf-8", errors="ignore")
    assert "st.session_state" in text, "frontend/app.py does not reference st.session_state"

    # Accept common patterns:
    # - if "messages" not in st.session_state: st.session_state.messages = []
    # - st.session_state["messages"] = []
    ok_init = (
        re.search(r'["\']messages["\']\s+not\s+in\s+st\.session_state', text)
        or "st.session_state.messages" in text
        or 'st.session_state["messages"]' in text
        or "st.session_state['messages']" in text
    )
    assert ok_init, "frontend/app.py does not appear to initialise/persist a messages list"


def test_m5_sidebar_shows_index_stats_loaded_at_startup():
    """
    Tests (Milestone 5 Acceptance):
    - Sidebar shows live index stats loaded at startup.

    Heuristic:
    - frontend/components/sidebar.py references /api/v1/index/stats OR "index stats".
    """
    sidebar_py = _frontend_dir / "components" / "sidebar.py"
    assert sidebar_py.exists(), "frontend/components/sidebar.py missing"

    text = sidebar_py.read_text(encoding="utf-8", errors="ignore").lower()
    assert ("/api/v1/index/stats" in text) or ("index/stats" in text) or ("index stats" in text), (
        "sidebar.py doesn't appear to load/display index stats"
    )


def test_m5_show_raw_query_toggle_exists_somewhere_in_frontend():
    """
    Tests (Milestone 5 Acceptance):
    - The UI has a "Show raw query" toggle that reveals the ES query used.

    Heuristic:
    - Look for the literal label "show raw query" OR checkbox controls that mention query.
    """
    assert _frontend_dir.exists(), "frontend/ directory missing"

    found = False
    for p in _frontend_dir.rglob("*.py"):
        t = p.read_text(encoding="utf-8", errors="ignore").lower()
        if "show raw query" in t:
            found = True
            break
        if "st.checkbox" in t and "query" in t:
            found = True
            break

    assert found, "Could not find a 'Show raw query' toggle/checkbox in frontend code"


def test_m5_blocked_query_is_user_friendly_no_traceback_rendering():
    """
    Tests (Milestone 5 Acceptance):
    - A blocked query displays a user-friendly message (not a stack trace).

    What we can check offline:
    - Frontend displays backend replies (response/content)
    - Frontend does not obviously render 'traceback' via st.write/st.markdown/st.code/etc.

    Note:
    - This is a heuristic static scan (not a runtime UI test).
    """
    assert _frontend_dir.exists(), "frontend/ directory missing"

    uses_response_key = False           # json["response"] or .get("response")
    uses_chat_message_ui = False        # st.chat_message(...)
    renders_content_field = False       # msg["content"] / get("content")
    appends_assistant_msg = False       # messages.append({"role":"assistant", ...})
    writes_traceback_to_ui = False      # suspicious UI printing

    ui_calls = ["st.write", "st.markdown", "st.code", "st.text", "st.error", "st.warning"]

    for p in _frontend_dir.rglob("*.py"):
        t = p.read_text(encoding="utf-8", errors="ignore")
        tl = t.lower()

        if ("['response']" in tl) or ('["response"]' in tl) or (".get('response')" in tl) or ('.get("response")' in tl):
            uses_response_key = True

        if "st.chat_message" in tl:
            uses_chat_message_ui = True

        if ("['content']" in tl) or ('["content"]' in tl) or (".get('content')" in tl) or ('.get("content")' in tl):
            renders_content_field = True

        if re.search(r"append\(\s*\{[^}]*['\"]role['\"]\s*:\s*['\"]assistant['\"]", tl):
            appends_assistant_msg = True

        if "traceback" in tl and any(call in tl for call in ui_calls):
            writes_traceback_to_ui = True

    shows_reply = uses_response_key or (uses_chat_message_ui and renders_content_field and appends_assistant_msg)
    assert shows_reply, (
        "Frontend doesn't appear to display backend replies. "
        "Expected either json['response'] usage, or chat UI pattern (st.chat_message + content + append assistant)."
    )
    assert not writes_traceback_to_ui, "Frontend appears to render tracebacks directly in the UI"