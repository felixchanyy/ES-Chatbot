# backend/services/query_safety.py

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

class SafetyStatus(Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    MODIFIED = "modified"

@dataclass
class ValidationResult:
    status: SafetyStatus
    query: Optional[Dict]
    reason: Optional[str]

class QuerySafetyLayer:
    def __init__(self):
        pass

    def validate(self, query: Dict) -> ValidationResult:
        """
        Runs all safety checks in order. Returns a ValidationResult.
        """
        # Placeholder: Implement validation logic
        return ValidationResult(
            status=SafetyStatus.ALLOWED,
            query=query,
            reason=None
        )