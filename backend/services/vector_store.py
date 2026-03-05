from elasticsearch import Elasticsearch
from langchain.schema import Document
from langchain_chroma import Chroma
from chromadb import HttpClient
from langchain_huggingface import HuggingFaceEmbeddings
from backend.services.es_client import ESClient
from config import settings


def store_docs(elastic_client: ESClient) -> list[Document]:
    docs = elastic_client.get_mapping()
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    client = HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port
    )

    vectorstore = Chroma(
        client=client,
        collection_name="gkg_mapping",
        embedding_function=embeddings
    )
    
    vectorstore.add_documents(docs)
