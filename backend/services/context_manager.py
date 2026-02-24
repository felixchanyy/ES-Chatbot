"""backend/services/context_manager.py

Milestone 4:
Shape Elasticsearch responses into compact, LLM-friendly payloads.

We keep this implementation dependency-free (no tiktoken) and use a
simple character-based budget.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from services.query_safety import ALWAYS_EXCLUDE_FIELDS


class ContextManager:
    """Shapes and truncates results to fit the LLM's context window."""

    def __init__(
        self,
        *,
        max_docs: int = 20,
        max_chars: int = 16000,
        always_exclude_fields: Optional[List[str]] = None,
    ) -> None:
        self.max_docs = int(max_docs)
        self.max_chars = int(max_chars)
        self.always_exclude_fields = list(
            ALWAYS_EXCLUDE_FIELDS if always_exclude_fields is None else always_exclude_fields
        )

    def shape_results(self, es_response: Dict[str, Any], query_type: str) -> Dict[str, Any]:
        """Return a compact representation of the ES response.

        query_type: 'aggregation' | 'retrieval'
        """
        if not isinstance(es_response, dict):
            return {"error": "es_response_not_a_dict"}

        if query_type == "aggregation":
            return self._shape_aggs(es_response)
        return self._shape_hits(es_response)

    def estimate_tokens(self, text: str) -> int:
        """Very rough token estimate (~4 chars per token)."""
        if not text:
            return 0
        return max(1, int(len(text) / 4))

    # -----------------
    # Internals
    # -----------------
    def _shape_aggs(self, es_response: Dict[str, Any]) -> Dict[str, Any]:
        aggs = es_response.get("aggregations") or es_response.get("aggs") or {}
        shaped = {
            "took_ms": es_response.get("took"),
            "timed_out": es_response.get("timed_out"),
            "total_hits": self._extract_total_hits(es_response.get("hits")),
            "aggregations": self._simplify_aggs_node(aggs),
        }

        return self._truncate_to_budget(shaped)

    def _shape_hits(self, es_response: Dict[str, Any]) -> Dict[str, Any]:
        hits_obj = es_response.get("hits") or {}
        hits = hits_obj.get("hits") or []
        docs: List[Dict[str, Any]] = []
        for h in hits[: self.max_docs]:
            if not isinstance(h, dict):
                continue
            src = h.get("_source")
            if isinstance(src, dict):
                docs.append(self._strip_toxic_fields(src))

        shaped = {
            "took_ms": es_response.get("took"),
            "timed_out": es_response.get("timed_out"),
            "total_hits": self._extract_total_hits(hits_obj),
            "documents": docs,
        }
        return self._truncate_to_budget(shaped)

    def _extract_total_hits(self, hits_obj: Any) -> Optional[int]:
        if not isinstance(hits_obj, dict):
            return None
        total = hits_obj.get("total")
        if isinstance(total, dict):
            v = total.get("value")
            return int(v) if isinstance(v, int) else None
        if isinstance(total, int):
            return total
        return None

    def _strip_toxic_fields(self, src: Dict[str, Any]) -> Dict[str, Any]:
        # ES _source.excludes should already remove these; this is a backstop.
        cleaned = dict(src)
        for f in self.always_exclude_fields:
            if f in cleaned:
                cleaned.pop(f, None)
        return cleaned

    def _simplify_aggs_node(self, node: Any) -> Any:
        """Simplify aggregation results recursively.

        Handles:
          - bucket aggs (terms / significant_terms / date_histogram): buckets
          - metric aggs: value
          - nested aggs
        """
        if not isinstance(node, dict):
            return node

        out: Dict[str, Any] = {}
        for name, body in node.items():
            if not isinstance(body, dict):
                out[name] = body
                continue

            if "buckets" in body and isinstance(body["buckets"], list):
                buckets_out = []
                for b in body["buckets"][:200]:
                    if not isinstance(b, dict):
                        continue
                    item: Dict[str, Any] = {
                        "key": b.get("key_as_string", b.get("key")),
                        "doc_count": b.get("doc_count"),
                    }
                    # include common metric sub-aggs
                    for k, v in b.items():
                        if k in ("key", "key_as_string", "doc_count"):
                            continue
                        if isinstance(v, dict) and "value" in v:
                            item[k] = v.get("value")
                        elif isinstance(v, dict) and "buckets" in v:
                            item[k] = self._simplify_aggs_node({k: v})[k]
                    buckets_out.append(item)
                out[name] = buckets_out
                continue

            if "value" in body and len(body.keys()) <= 3:
                out[name] = body.get("value")
                continue

            nested = body.get("aggregations") or body.get("aggs")
            if isinstance(nested, dict):
                out[name] = self._simplify_aggs_node(nested)
            else:
                out[name] = body

        return out

    def _truncate_to_budget(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure JSON-serialised payload stays within a rough character budget."""
        try:
            s = json.dumps(payload, ensure_ascii=False)
        except Exception:
            return payload

        if len(s) <= self.max_chars:
            return payload

        # First, truncate docs or buckets if present.
        if "documents" in payload and isinstance(payload["documents"], list):
            docs = payload["documents"]
            while docs and len(json.dumps(payload, ensure_ascii=False)) > self.max_chars:
                docs.pop()
            payload["documents_truncated"] = True
            return payload

        if "aggregations" in payload and isinstance(payload["aggregations"], dict):
            # aggs = payload["aggregations"]
            # for k, v in aggs.items():
            #     if isinstance(v, list):
            #         while v and len(json.dumps(payload, ensure_ascii=False)) > self.max_chars:
            #             v.pop()
            #         aggs[k] = v
            # payload["aggregations_truncated"] = True
            return payload

        # Last resort: minimal envelope.
        return {
            "took_ms": payload.get("took_ms"),
            "timed_out": payload.get("timed_out"),
            "total_hits": payload.get("total_hits"),
            "note": "results_truncated_due_to_context_budget",
        }
