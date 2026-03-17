from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Elasticsearch
    es_host: str
    es_username: str
    es_password: str
    es_index: str = "gkg"
    es_verify_ssl: bool = False
    es_request_timeout_seconds: int = 30

    # LLM
    llm_base_url: str
    llm_model_name: str
    llm_api_key: str = "not-required"
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 60

    # Chroma / schema retrieval
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_collection: str = "gkg_mapping"
    schema_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    schema_search_k: int = 8

    # Agent / retry controls
    agent_max_iterations: int = 30
    query_max_attempts: int = 4
    max_result_docs: int = 20
    max_agg_buckets: int = 100
    max_validation_candidates: int = 50


settings = Settings()