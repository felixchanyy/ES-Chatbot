from __future__ import annotations

from typing import Any, Dict

from elasticsearch import AsyncElasticsearch

from config import settings


class ESClient:
    """Thin wrapper around AsyncElasticsearch with response normalisation."""

    def __init__(self) -> None:
        self.client = AsyncElasticsearch(
            hosts=[settings.es_host],
            basic_auth=(settings.es_username, settings.es_password),
            verify_certs=settings.es_verify_ssl,
            request_timeout=settings.es_request_timeout_seconds,
        )
        
    async def ping(self) -> bool:
        """Ping the cluster to check health using the existing connection pool."""
        return await self.client.ping()

    @staticmethod
    def _to_dict(response: Any) -> Dict[str, Any]:
        return response.body if hasattr(response, "body") else response

    async def search(self, index: str, query: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.client.search(index=index, body=query)
        return self._to_dict(response)

    async def get_index_mapping(self, index: str | None = None) -> Dict[str, Any]:
        response = await self.client.indices.get_mapping(index=index or settings.es_index)
        return self._to_dict(response)

    async def get_index_stats(self) -> Dict[str, Any]:
        count_resp = self._to_dict(await self.client.count(index=settings.es_index))
        total_docs = count_resp.get("count", 0)

        stats_resp = self._to_dict(
            await self.client.indices.stats(index=settings.es_index, metric="store")
        )
        index_data = stats_resp.get("indices", {}).get(settings.es_index, {})
        size_bytes = index_data.get("total", {}).get("store", {}).get("size_in_bytes", 0)

        agg_resp = self._to_dict(
            await self.client.search(
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
        )

        aggs = agg_resp.get("aggregations", {})
        return {
            "total_documents": total_docs,
            "index_size_bytes": size_bytes,
            "earliest_date": aggs.get("earliest_date", {}).get("value_as_string"),
            "latest_date": aggs.get("latest_date", {}).get("value_as_string"),
            "top_sources": [
                {"source": bucket["key"], "count": bucket["doc_count"]}
                for bucket in aggs.get("top_sources", {}).get("buckets", [])
            ],
        }
            "earliest_date": earliest,
            "latest_date": latest,
            "top_sources": top_sources,
        }
    
# Create a single global instance
es_client = ESClient()
