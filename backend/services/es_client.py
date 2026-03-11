# backend/services/es_client.py

import hashlib
from typing import Any, Dict, List

from elasticsearch import AsyncElasticsearch
from langchain_core.documents import Document
from config import settings

class ESClient:
    """
    Elasticsearch client wrapper.
    """

    def __init__(self):
        self.client = AsyncElasticsearch(
            hosts=[settings.es_host],
            basic_auth=(settings.es_username, settings.es_password),
            verify_certs=settings.es_verify_ssl
        )

    async def search(self, index: str, query: dict):
        response = await self.client.search(index=index, body=query)

        # Elasticsearch 8.x returns a Response object
        # Ensure we always return a plain dict
        if hasattr(response, "body"):
            return response.body

        return response
    
    @staticmethod
    def _stable_id(value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()
    
    async def get_index_stats(self) -> dict:
        """
        Return summary statistics about the index:
        - total_documents
        - index_size_bytes
        - earliest_date
        - latest_date
        - top_sources
        """

        # 1️⃣ Document count
        count_resp = await self.client.count(index=settings.es_index)
        total_docs = count_resp.get("count", 0)

        # 2️⃣ Index size
        stats_resp = await self.client.indices.stats(
            index=settings.es_index,
            metric="store",
        )

        index_data = stats_resp["indices"].get(settings.es_index, {})
        size_bytes = (
            index_data.get("total", {})
            .get("store", {})
            .get("size_in_bytes", 0)
        )

        # 3️⃣ Aggregations for earliest/latest + top sources
        agg_resp = await self.client.search(
            index=settings.es_index,
            size=0,
            track_total_hits=False,
            aggs={
                "earliest_date": {"min": {"field": "V21Date"}},
                "latest_date": {"max": {"field": "V21Date"}},
                "top_sources": {
                    "terms": {
                        "field": "V2SrcCmnName.V2SrcCmnName.keyword",
                        "size": 10,
                    }
                },
            },
        )

        aggs = agg_resp.get("aggregations", {})

        earliest = aggs.get("earliest_date", {}).get("value_as_string")
        latest = aggs.get("latest_date", {}).get("value_as_string")

        buckets = aggs.get("top_sources", {}).get("buckets", [])
        top_sources = [
            {"source": bucket["key"], "count": bucket["doc_count"]}
            for bucket in buckets
        ]

        return {
            "total_documents": total_docs,
            "index_size_bytes": size_bytes,
            "earliest_date": earliest,
            "latest_date": latest,
            "top_sources": top_sources,
        }
    
    async def get_chunks(self) -> list[Document]:
        mapping = await self.client.indices.get_mapping(index=settings.es_index)
        index_payload = mapping.get(settings.es_index)
        if index_payload is None and mapping:
            index_payload = next(iter(mapping.values()))
        if not isinstance(index_payload, dict):
            return []

        mappings = index_payload.get("mappings", {})
        properties = mappings.get("properties", {})
        chunks: List[Dict[str, Any]] = []

        overview = (
            f"Index {settings.es_index} schema overview. "
            "Use only fields that appear in this schema. "
            "For terms aggregations prefer keyword fields or keyword subfields when available."
        )
        chunks.append(
            {
                "id": self._stable_id("overview"),
                "document": overview,
                "metadata": {"kind": "overview", "index": settings.es_index},
            }
        )

        def walk(node: Dict[str, Any], prefix: str = "") -> None:
            for field_name, spec in node.items():
                if not isinstance(spec, dict):
                    continue

                full_name = f"{prefix}.{field_name}" if prefix else field_name
                field_type = spec.get("type") or ("object" if "properties" in spec else "unknown")
                subfields = spec.get("fields", {}) if isinstance(spec.get("fields"), dict) else {}
                subfield_names = list(subfields.keys())

                usage_notes: List[str] = []
                if field_type == "date":
                    usage_notes.append("Use this field for date ranges and time filtering")
                elif field_type == "keyword":
                    usage_notes.append("Use this field for exact matches, filters, sorting, and terms aggregations")
                elif field_type == "text":
                    usage_notes.append("Use this field for full text matching")
                    if "keyword" in subfields:
                        usage_notes.append(
                            f"Use {full_name}.keyword for exact matching and terms aggregations"
                        )
                elif field_type in {"integer", "long", "float", "double", "scaled_float", "short", "byte"}:
                    usage_notes.append("Use this field for numeric ranges, sorting, or statistical aggregations")
                elif field_type == "boolean":
                    usage_notes.append("Use this field as a boolean filter")
                elif field_type in {"nested", "object"}:
                    usage_notes.append("This is a container field that may have child properties")

                text_parts = [
                    f"Index: {settings.es_index}",
                    f"Field: {full_name}",
                    f"Type: {field_type}",
                ]
                if subfield_names:
                    text_parts.append(
                        f"Subfields: {', '.join(f'{full_name}.{name}' for name in subfield_names)}"
                    )
                if usage_notes:
                    text_parts.append(f"Usage: {'; '.join(usage_notes)}")

                chunks.append(
                    {
                        "id": self._stable_id(full_name),
                        "document": ". ".join(text_parts),
                        "metadata": {
                            "kind": "field",
                            "field": full_name,
                            "type": field_type,
                            "index": settings.es_index,
                        },
                    }
                )

                for subfield_name, sub_spec in subfields.items():
                    sub_type = sub_spec.get("type", "unknown") if isinstance(sub_spec, dict) else "unknown"
                    subfield_path = f"{full_name}.{subfield_name}"
                    chunks.append(
                        {
                            "id": self._stable_id(subfield_path),
                            "document": (
                                f"Index: {settings.es_index}. Field: {subfield_path}. Type: {sub_type}. "
                                "Usage: Use this field for exact matches, filters, sorting, and terms aggregations."
                            ),
                            "metadata": {
                                "kind": "subfield",
                                "field": subfield_path,
                                "type": sub_type,
                                "index": settings.es_index,
                            },
                        }
                    )

                child_properties = spec.get("properties")
                if isinstance(child_properties, dict):
                    walk(child_properties, full_name)

        if isinstance(properties, dict):
            walk(properties)

        return chunks
    