# backend/services/es_client.py

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
    
    async def get_mapping(self) -> dict:
        mapping = await self.client.indices.get_mapping(index=settings.es_index)
        properties = mapping[settings.es_index]["mappings"]["properties"]

        docs = []
        for field, info in properties.items():
            field_type = info.get("type", "object")

            text = f"""
            Field: {field}
            Type: {field_type}
            Description:  {info}
            """
            docs.append(Document(page_content=text))

        return docs