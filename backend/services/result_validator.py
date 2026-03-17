from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from config import settings


@dataclass
class ValidationConstraints:
    requested_count: Optional[int] = None
    entity_type: Optional[str] = None
    category: Optional[str] = None
    is_ranking: bool = False


@dataclass
class ValidationOutcome:
    passed: bool
    reason: str
    requested_count: int = 0
    valid_count: int = 0
    valid_items: List[Dict[str, Any]] = field(default_factory=list)
    invalid_items: List[Dict[str, Any]] = field(default_factory=list)
    invalid_names: List[str] = field(default_factory=list)
    refinement_hint: str = ""


class ResultValidator:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model_name,
            temperature=0,
            timeout=settings.llm_timeout_seconds,
        )

        self.location_denylist = {
            "Los Angeles",
            "Las Vegas",
            "New York",
            "San Francisco",
            "Abu Dhabi",
            "Middle East",
            "Iran",
            "Israel",
            "United States",
            "United Kingdom",
            "Asia",
            "Europe",
            "Indian Ocean",
            "South Korea",
            "North Korea",
            "China",
            "Russia",
            "Ukraine",
            "Singapore",
            "Washington",
            "California",
            "London",
            "Paris",
            "Tokyo",
        }

    def extract_constraints(self, question: str) -> ValidationConstraints:
        q = question.lower()
        requested_count = None
        match = re.search(r"\btop\s+(\d+)\b", q)
        if match:
            requested_count = int(match.group(1))

        entity_type = None
        if "people" in q or "person" in q or "individual" in q:
            entity_type = "person"
        elif "organisation" in q or "organization" in q or "org" in q:
            entity_type = "organization"
        elif "country" in q or "countries" in q or "location" in q or "locations" in q:
            entity_type = "location"
        elif "source" in q or "domain" in q:
            entity_type = "source"
        elif "theme" in q or "topic" in q:
            entity_type = "theme"

        category = None
        person_category_patterns = [
            r"top\s+\d+\s+([a-z\- ]+?)\s+people",
            r"top\s+\d+\s+([a-z\- ]+?)\s+persons",
            r"most mentioned\s+([a-z\- ]+?)\s+people",
        ]
        for pattern in person_category_patterns:
            match = re.search(pattern, q)
            if match:
                category = match.group(1).strip()
                break

        is_ranking = any(term in q for term in ["top ", "most mentioned", "most common", "ranking"])
        return ValidationConstraints(
            requested_count=requested_count,
            entity_type=entity_type,
            category=category,
            is_ranking=is_ranking,
        )

    def _looks_like_person_basic(self, text: str) -> bool:
        if not text:
            return False
        if text in self.location_denylist:
            return False

        parts = [p for p in re.split(r"\s+", text) if p]
        if len(parts) < 2 or len(parts) > 5:
            return False

        alpha_parts = [p for p in parts if any(ch.isalpha() for ch in p)]
        if len(alpha_parts) < 2:
            return False

        if not all(p[:1].isupper() for p in alpha_parts):
            return False

        return True

    async def _classify_person_names(self, names: List[str]) -> Dict[str, bool]:
        if not names:
            return {}

        prompt = {
            "task": "classify_person_names",
            "entities": names,
            "instructions": [
                "Return strict JSON only.",
                "For each entity, decide if it is a real person's name.",
                "Locations, countries, cities, regions, organizations, events, or topics must be false.",
                "If uncertain, return false.",
                "Output format: {\"results\":[{\"name\":...,\"is_person\":true|false,\"reason\":...}]}",
            ],
        }

        response = await asyncio.to_thread(
            self.llm.invoke,
            [
                {"role": "system", "content": "You are a strict entity classifier. Return JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )

        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)

        try:
            match = re.search(r"\{.*\}", str(content), re.DOTALL)
            data = json.loads(match.group(0) if match else str(content))
        except Exception:
            return {name: False for name in names}

        results = data.get("results", []) if isinstance(data, dict) else []
        mapped: Dict[str, bool] = {}
        for item in results:
            if isinstance(item, dict) and item.get("name"):
                mapped[str(item["name"])] = bool(item.get("is_person", False))
        return {name: mapped.get(name, False) for name in names}

    async def _classify_category_matches(self, names: List[str], category: str, entity_type: str) -> Dict[str, bool]:
        if not names:
            return {}

        prompt = {
            "task": "classify_entity_category_match",
            "entity_type": entity_type,
            "requested_category": category,
            "entities": names,
            "instructions": [
                "Return strict JSON only.",
                "For each entity, decide whether it satisfies the requested category.",
                "If uncertain, return false.",
                "Output format: {\"results\":[{\"name\":...,\"match\":true|false,\"reason\":...}]}",
            ],
        }

        response = await asyncio.to_thread(
            self.llm.invoke,
            [
                {"role": "system", "content": "You are a strict entity classifier. Return JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )

        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)

        try:
            match = re.search(r"\{.*\}", str(content), re.DOTALL)
            data = json.loads(match.group(0) if match else str(content))
        except Exception:
            return {name: False for name in names}

        results = data.get("results", []) if isinstance(data, dict) else []
        mapped: Dict[str, bool] = {}
        for item in results:
            if isinstance(item, dict) and item.get("name"):
                mapped[str(item["name"])] = bool(item.get("match", False))
        return {name: mapped.get(name, False) for name in names}

    async def validate(
        self,
        *,
        question: str,
        shaped_results: Dict[str, Any],
    ) -> ValidationOutcome:
        constraints = self.extract_constraints(question)
        aggs = shaped_results.get("aggregations") or {}
        if not isinstance(aggs, dict) or not aggs:
            total_hits = shaped_results.get("total_hits")
            passed = bool(total_hits) and not constraints.is_ranking
            return ValidationOutcome(
                passed=passed,
                reason="no_aggregations" if constraints.is_ranking else "document_results_only",
                refinement_hint="Switch to an aggregation query for ranking requests." if constraints.is_ranking else "",
            )

        first_agg_name = next(iter(aggs.keys()))
        buckets = aggs.get(first_agg_name) or []
        if not isinstance(buckets, list):
            return ValidationOutcome(
                False,
                "aggregation_not_a_bucket_list",
                refinement_hint="Return buckets as a terms aggregation.",
            )

        candidates = buckets[: settings.max_validation_candidates]
        valid_items: List[Dict[str, Any]] = []
        invalid_items: List[Dict[str, Any]] = []

        if constraints.entity_type == "person":
            prelim_persons = []
            prelim_non_persons = []

            for bucket in candidates:
                key = str(bucket.get("key", "")).strip()
                if self._looks_like_person_basic(key):
                    prelim_persons.append(bucket)
                else:
                    prelim_non_persons.append(bucket)

            valid_items.extend(prelim_persons)
            invalid_items.extend(prelim_non_persons)

            names = [str(item.get("key", "")).strip() for item in valid_items]
            llm_map = await self._classify_person_names(names)

            rechecked_valid = []
            for item in valid_items:
                key = str(item.get("key", "")).strip()
                if llm_map.get(key, False):
                    rechecked_valid.append(item)
                else:
                    invalid_items.append(item)

            valid_items = rechecked_valid

        else:
            valid_items = list(candidates)

        if constraints.category and constraints.entity_type == "person":
            names = [str(item.get("key", "")).strip() for item in valid_items]
            category_map = await self._classify_category_matches(
                names, constraints.category, constraints.entity_type
            )

            rechecked_valid = []
            for item in valid_items:
                key = str(item.get("key", "")).strip()
                if category_map.get(key, False):
                    rechecked_valid.append(item)
                else:
                    invalid_items.append(item)

            valid_items = rechecked_valid

        requested = constraints.requested_count or 0
        passed = True
        reason = "validated"
        refinement_hint = ""

        if constraints.is_ranking and requested > 0:
            if len(valid_items) >= requested:
                valid_items = valid_items[:requested]
                passed = True
                reason = "enough_valid_ranked_items"
            else:
                passed = False
                reason = f"only_{len(valid_items)}_valid_items_for_top_{requested}"
                refinement_hint = (
                    "Broaden candidate pool, exclude invalid buckets already seen, and tighten person/category constraints. "
                    "Do not stop until enough valid entities are obtained or attempts are exhausted."
                )

        invalid_names = [str(item.get("key", "")).strip() for item in invalid_items if item.get("key")]

        return ValidationOutcome(
            passed=passed,
            reason=reason,
            requested_count=requested,
            valid_count=len(valid_items),
            valid_items=valid_items,
            invalid_items=invalid_items,
            invalid_names=invalid_names,
            refinement_hint=refinement_hint,
        )