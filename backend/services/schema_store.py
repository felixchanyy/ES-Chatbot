from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from services.es_client import ESClient

logger = logging.getLogger(__name__)


class SchemaStore:
    """Synchronise the Elasticsearch mapping into Chroma for schema retrieval."""

    def __init__(self) -> None:
        self.es = ESClient()
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.schema_embedding_model)
        self.client: Optional[chromadb.HttpClient] = None
        self.collection = None

    def _ensure_client_and_collection(self) -> None:
        if self.client is None:
            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                ssl=settings.chroma_ssl,
            )
        if self.collection is None:
            self.collection = self.client.get_or_create_collection(
                name=settings.chroma_collection,
                metadata={"source": "elasticsearch_schema", "index": settings.es_index},
            )

    async def ensure_schema_collection_synced(self, force: bool = False) -> None:
        await asyncio.to_thread(self._ensure_client_and_collection)

        count = await asyncio.to_thread(self.collection.count)
        if count > 0 and not force:
            return

        mapping = await self.es.get_index_mapping(settings.es_index)
        chunks = self._mapping_to_chunks(mapping)
        if not chunks:
            raise RuntimeError("Elasticsearch mapping sync produced no schema chunks")

        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["document"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        embeddings = await asyncio.to_thread(self.embeddings.embed_documents, documents)

        existing = await asyncio.to_thread(self.collection.get, include=[])
        existing_ids = set(existing.get("ids", []))
        stale_ids = list(existing_ids - set(ids))
        if stale_ids:
            await asyncio.to_thread(self.collection.delete, ids=stale_ids)

        await asyncio.to_thread(
            self.collection.upsert,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("schema_synced_to_chroma", extra={"result_count": len(documents)})

    async def search_schema(self, question: str, k: int | None = None) -> List[str]:
        await self.ensure_schema_collection_synced(force=False)
        query_embedding = await asyncio.to_thread(self.embeddings.embed_query, question)
        result = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=k or settings.schema_search_k,
            include=["documents", "metadatas"],
        )
        documents = result.get("documents", [[]])
        return documents[0] if documents else []

    async def get_schema_overview(self, limit: int = 12) -> List[str]:
        await self.ensure_schema_collection_synced(force=False)
        result = await asyncio.to_thread(self.collection.get, limit=limit, include=["documents"])
        return result.get("documents", []) or []

    def _mapping_to_chunks(self, mapping_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        index_payload = mapping_response.get(settings.es_index)
        if index_payload is None and mapping_response:
            index_payload = next(iter(mapping_response.values()))
        if not isinstance(index_payload, dict):
            return []

        mappings = index_payload.get("mappings", {})
        properties = mappings.get("properties", {})
        chunks: List[Dict[str, Any]] = []

        chunks.append(
            {
                "id": self._stable_id("overview"),
                "document": (
                    f"Index {settings.es_index} schema overview. Use only fields that appear in this schema. "
                    "For terms aggregations prefer keyword fields or keyword subfields when available."
                ),
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
                        usage_notes.append(f"Use {full_name}.keyword for exact matching and terms aggregations")
                elif field_type in {"integer", "long", "float", "double", "scaled_float", "short", "byte"}:
                    usage_notes.append("Use this field for numeric ranges, sorting, or statistical aggregations")
                elif field_type == "boolean":
                    usage_notes.append("Use this field as a boolean filter")
                elif field_type in {"nested", "object"}:
                    usage_notes.append("This is a container field that may have child properties")

                parts = [
                    f"Index: {settings.es_index}",
                    f"Field: {full_name}",
                    f"Type: {field_type}",
                ]
                if subfield_names:
                    parts.append(
                        f"Subfields: {', '.join(f'{full_name}.{name}' for name in subfield_names)}"
                    )
                if usage_notes:
                    parts.append(f"Usage: {'; '.join(usage_notes)}")

                chunks.append(
                    {
                        "id": self._stable_id(full_name),
                        "document": ". ".join(parts),
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

    @staticmethod
    def _stable_id(value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()


_schema_store_singleton: Optional[SchemaStore] = None


def get_schema_store() -> SchemaStore:
    global _schema_store_singleton
    if _schema_store_singleton is None:
        _schema_store_singleton = SchemaStore()
    return _schema_store_singleton