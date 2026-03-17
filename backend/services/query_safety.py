from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


ALLOWED_TOP_LEVEL_KEYS = {
    "query",
    "aggs",
    "aggregations",
    "size",
    "from",
    "sort",
    "_source",
    "highlight",
    "track_total_hits",
    "search_after",
    "pit",
}

ALWAYS_EXCLUDE_FIELDS = [
    "event.original",
    "V2GCAM.DictionaryDimId",
    "log",
    "filename",
    "filename_path",
    "host",
    "@version",
]


class SafetyStatus(str, Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    MODIFIED = "modified"


@dataclass
class ValidationResult:
    status: SafetyStatus
    query: Optional[Dict[str, Any]]
    reason: Optional[str]


class QuerySafetyLayer:
    def __init__(
        self,
        *,
        max_result_docs: int | None = None,
        max_agg_buckets: int | None = None,
        always_exclude_fields: list[str] | None = None,
    ) -> None:
        self.max_result_docs = int(max_result_docs or self._get_default_max_result_docs())
        self.max_agg_buckets = int(max_agg_buckets or self._get_default_max_agg_buckets())
        self.always_exclude_fields = list(
            ALWAYS_EXCLUDE_FIELDS if always_exclude_fields is None else always_exclude_fields
        )

    def _get_default_max_result_docs(self) -> int:
        try:
            from config import settings

            return int(settings.max_result_docs)
        except Exception:
            return 20

    def _get_default_max_agg_buckets(self) -> int:
        try:
            from config import settings

            return int(settings.max_agg_buckets)
        except Exception:
            return 100

    def validate(self, query: Dict[str, Any]) -> ValidationResult:
        if not isinstance(query, dict):
            return ValidationResult(SafetyStatus.BLOCKED, None, "query_not_a_dict")

        for key in query.keys():
            if key not in ALLOWED_TOP_LEVEL_KEYS:
                return ValidationResult(SafetyStatus.BLOCKED, None, f"invalid_top_level_key:{key}")

        if self._contains_script(query):
            return ValidationResult(SafetyStatus.BLOCKED, None, "script_detected")

        modified_reasons: list[str] = []
        if self._enforce_size_cap(query, self.max_result_docs):
            modified_reasons.append(f"size_capped_to_{self.max_result_docs}")
        if self._inject_source_excludes(query, self.always_exclude_fields):
            modified_reasons.append("source_excludes_injected")
        if self._cap_agg_bucket_sizes(query, self.max_agg_buckets):
            modified_reasons.append(f"agg_buckets_capped_to_{self.max_agg_buckets}")

        if modified_reasons:
            return ValidationResult(SafetyStatus.MODIFIED, query, ";".join(modified_reasons))
        return ValidationResult(SafetyStatus.ALLOWED, query, None)

    def _contains_script(self, node: Any) -> bool:
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() == "script":
                    return True
                if self._contains_script(value):
                    return True
        elif isinstance(node, list):
            return any(self._contains_script(item) for item in node)
        return False

    def _enforce_size_cap(self, query: Dict[str, Any], max_docs: int) -> bool:
        if "size" not in query:
            return False
        try:
            current = int(query["size"])
        except Exception:
            query["size"] = max_docs
            return True
        if current > max_docs:
            query["size"] = max_docs
            return True
        return False

    def _inject_source_excludes(self, query: Dict[str, Any], excludes: list[str]) -> bool:
        existing = query.get("_source")
        if existing is None:
            query["_source"] = {"excludes": list(excludes)}
            return True
        if isinstance(existing, bool):
            query["_source"] = {"excludes": list(excludes)}
            return True
        if isinstance(existing, list):
            query["_source"] = {"includes": existing, "excludes": list(excludes)}
            return True
        if isinstance(existing, dict):
            current_excludes = existing.get("excludes")
            if current_excludes is None:
                existing["excludes"] = list(excludes)
                return True
            if isinstance(current_excludes, str):
                current_list = [current_excludes]
            elif isinstance(current_excludes, list):
                current_list = [str(x) for x in current_excludes]
            else:
                current_list = [str(current_excludes)]
            merged = list(dict.fromkeys(current_list + list(excludes)))
            changed = merged != current_list
            existing["excludes"] = merged
            query["_source"] = existing
            return changed
        query["_source"] = {"excludes": list(excludes)}
        return True

    def _cap_agg_bucket_sizes(self, query: Dict[str, Any], max_buckets: int) -> bool:
        modified = False
        aggs = query.get("aggs") or query.get("aggregations")
        if not isinstance(aggs, dict):
            return False

        def walk(aggs_node: Dict[str, Any]) -> None:
            nonlocal modified
            for _, body in aggs_node.items():
                if not isinstance(body, dict):
                    continue
                for agg_type in ("terms", "significant_terms"):
                    if agg_type in body and isinstance(body[agg_type], dict):
                        params = body[agg_type]
                        if "size" in params:
                            try:
                                size_val = int(params["size"])
                            except Exception:
                                params["size"] = max_buckets
                                modified = True
                            else:
                                if size_val > max_buckets:
                                    params["size"] = max_buckets
                                    modified = True
                sub = body.get("aggs") or body.get("aggregations")
                if isinstance(sub, dict):
                    walk(sub)

        walk(aggs)
        return modified