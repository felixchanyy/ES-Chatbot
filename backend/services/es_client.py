# backend/services/es_client.py

from elasticsearch import AsyncElasticsearch
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

    async def search(self, query: dict) -> dict:
        """
        Execute a search query against Elasticsearch.
        """
        response = await self.client.search(
            index=settings.es_index,
            body=query
        )
        return response

    async def get_index_stats(self) -> dict:
        """
        Return summary statistics about the index.
        """
        # Placeholder: Implement index stats logic
        return {
            "total_documents": 0,
            "index_size_bytes": 0,
            "earliest_date": "",
            "latest_date": "",
            "top_sources": []
        }