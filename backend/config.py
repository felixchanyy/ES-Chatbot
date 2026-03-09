# backend/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
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

    # ChromaDB
    chroma_host: str = "chromadb"
    chroma_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env"
    )

settings = Settings()