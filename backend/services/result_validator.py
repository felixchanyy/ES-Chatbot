# backend/services/result_validator.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RequestConstraints:
    requested_count: Optional[int] = None
    entity_type: Optional[str] = None   # e.g. "person"
    is_ranking: bool = False


@dataclass
class ValidationOutcome:
    is_sufficient: bool
    valid_items: List[Dict[str, Any]]
    invalid_items: List[Dict[str, Any]]
    reason: str


class ResultValidator:
    def extract_constraints(self, question: str) -> RequestConstraints:
        q = question.lower()

        requested_count = None
        m = re.search(r"\btop\s+(\d+)\b", q)
        if m:
            requested_count = int(m.group(1))

        entity_type = None
        if "people" in q or "persons" in q or "person" in q:
            entity_type = "person"

        is_ranking = "top " in q or "most mentioned" in q or "most common" in q

        return RequestConstraints(
            requested_count=requested_count,
            entity_type=entity_type,
            is_ranking=is_ranking,
        )

    def validate_aggregation(
        self,
        question: str,
        shaped_results: Dict[str, Any],
    ) -> ValidationOutcome:
        constraints = self.extract_constraints(question)
        aggs = shaped_results.get("aggregations") or {}

        if not isinstance(aggs, dict) or not aggs:
            return ValidationOutcome(False, [], [], "no_aggregations")

        first_agg_name = next(iter(aggs.keys()))
        buckets = aggs.get(first_agg_name) or []
        if not isinstance(buckets, list):
            return ValidationOutcome(False, [], [], "aggregation_not_a_bucket_list")

        valid_items = []
        invalid_items = []

        for bucket in buckets:
            key = str(bucket.get("key", "")).strip()
            if constraints.entity_type == "person":
                if self._looks_like_person(key):
                    valid_items.append(bucket)
                else:
                    invalid_items.append(bucket)
            else:
                valid_items.append(bucket)

        requested = constraints.requested_count or 0

        if constraints.is_ranking and requested > 0:
            if len(valid_items) >= requested:
                return ValidationOutcome(True, valid_items[:requested], invalid_items, "enough_valid_ranked_items")
            return ValidationOutcome(
                False,
                valid_items,
                invalid_items,
                f"only_{len(valid_items)}_valid_items_for_top_{requested}",
            )

        return ValidationOutcome(bool(valid_items), valid_items, invalid_items, "generic_validation")

    def _looks_like_person(self, text: str) -> bool:
        # Lightweight heuristic. Replace with stronger classifier later.
        if not text:
            return False

        banned_exact = {
            "Abu Dhabi",
            "Indian Ocean",
            "Middle East",
            "United States",
            "Israel",
            "Iran",
        }
        if text in banned_exact:
            return False

        # crude person-like heuristic: 2-4 title-case tokens
        parts = [p for p in text.split() if p]
        if len(parts) < 2 or len(parts) > 4:
            return False

        if all(p[:1].isupper() for p in parts if p[:1].isalpha()):
            return True

        return False