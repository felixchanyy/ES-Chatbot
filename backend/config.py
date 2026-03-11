from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env.

    Notes:
    - CHROMA_PORT defaults to the container's internal port (8000), not the host-mapped port.
    - QUERY_MAX_ATTEMPTS controls the multi-stage retry loop in /chat.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Elasticsearch
    es_host: str
    es_username: str
    es_password: str
    es_index: str = "gkg"
    es_verify_ssl: bool = False

    # LLM
    llm_base_url: str
    llm_model_name: str
    llm_api_key: str = "not-required"

    # Safety
    max_result_docs: int = 20
    max_agg_buckets: int = 50

    # ChromaDB / schema retrieval
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_collection: str = "gkg_mapping"
    schema_embedding_model: str = "all-MiniLM-L6-v2"

    # Multi-stage querying
    query_max_attempts: int = 3


settings = Settings()