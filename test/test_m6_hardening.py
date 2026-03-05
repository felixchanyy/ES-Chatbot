"""
Milestone 6 — Hardening & Documentation

What this file tests (offline):
- README exists and includes onboarding essentials (heuristic keyword check)
- Chat input length validation exists in backend/models/schemas.py
- (Optional) basic structured logging hints exist (heuristic; won't fail if you don't want)

Adjust assertions to match your final hardening implementation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


# -----------------------------
# Path bootstrap
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../ES-Chatbot
_backend_dir = PROJECT_ROOT / "backend"
for p in (PROJECT_ROOT, _backend_dir):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def test_m6_readme_exists_and_has_onboarding_sections():
    """
    Tests (Milestone 6 Acceptance):
    - README.md exists at repo root
    - README contains enough onboarding info for a new developer:
        setup / installation
        architecture overview
        how to run tests
        known limitations / out of scope

    This is a heuristic keyword check (fast + offline).
    """
    readme = PROJECT_ROOT / "README.md"
    assert readme.exists(), "README.md is missing at repo root"

    text = readme.read_text(encoding="utf-8", errors="ignore").lower()

    must_have_any = {
        "setup": ["setup", "installation", "requirements", ".env", "docker compose", "docker-compose"],
        "architecture": ["architecture", "system architecture", "data flow", "component"],
        "tests": ["pytest", "run tests", "testing"],
        "limitations": ["known limitations", "limitations", "out of scope"],
    }

    missing = []
    for section, keywords in must_have_any.items():
        if not any(k in text for k in keywords):
            missing.append(section)

    assert not missing, f"README seems to be missing onboarding sections: {missing}"


def test_m6_chat_request_has_input_length_validation():
    """
    Tests (Milestone 6 Task):
    - Add input length validation to chat endpoint schemas.

    Expected (from spec):
    - ChatRequest.message has min_length=1, max_length=1000
    - ChatRequest.history has max_length=20 (optional but recommended)

    This test scans backend/models/schemas.py for Pydantic Field constraints.
    """
    schemas = PROJECT_ROOT / "backend" / "models" / "schemas.py"
    assert schemas.exists(), "backend/models/schemas.py missing"

    text = schemas.read_text(encoding="utf-8", errors="ignore")

    # message: str = Field(..., min_length=1, max_length=1000)
    assert re.search(r"message\s*:\s*str\s*=\s*Field\([^)]*min_length\s*=\s*1", text), (
        "ChatRequest.message min_length=1 not found"
    )
    assert re.search(r"message\s*:\s*str\s*=\s*Field\([^)]*max_length\s*=\s*1000", text), (
        "ChatRequest.message max_length=1000 not found"
    )

    # history max_length=20 (either in Field(...) or list[...] = Field(..., max_length=20)
    assert ("max_length=20" in text) or ("max_items=20" in text), (
        "Expected some history length cap (max_length=20 or max_items=20) not found"
    )


def test_m6_no_hardcoded_credentials_in_backend_config():
    """
    Tests (Hardening / hygiene):
    - Backend config should not hardcode real credentials (simple heuristic).

    This test will FAIL if you literally keep:
      es_username="elastic" and es_password="changeme" hardcoded as defaults.

    If you intentionally allow placeholders in .env.example but not in code,
    then keep this check.

    If your design allows safe placeholders, adjust or remove this test.
    """
    config_py = PROJECT_ROOT / "backend" / "config.py"
    if not config_py.exists():
        pytest.skip("backend/config.py not found (project structure differs)")

    text = config_py.read_text(encoding="utf-8", errors="ignore").lower()

    # Heuristic: defaults like 'changeme' should not be in runtime code
    assert "changeme" not in text, "Found 'changeme' in backend/config.py (avoid hardcoded passwords)"