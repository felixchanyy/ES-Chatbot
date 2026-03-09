from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb import HttpClient
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from services.es_client import ESClient
from config import settings

logger = logging.getLogger(__name__)

class ChromaClient:
    """
    Client for interacting with ChromaDB vector store.
    This is used to store and retrieve vector embeddings for gdelt metadata fields.
    """
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.collection_name = "gkg"
        client = HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

        self.vectorstore: Chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            client=client
        )

    async def initialize_data(self, es_client: ESClient):
        """
        Call this method immediately after instantiating the class to load initial data.
        """
        if es_client and self.vectorstore._collection.count() == 0:
            await self.add_documents(es_client)

    async def add_documents(self, es_client: ESClient):
        """
        Add a list of langchain.schema.Document to Chroma and persist.
        """
        docs = await es_client.get_mapping()
        self.vectorstore.add_documents(docs)
        logger.info(f"Added {len(docs)} documents to collection '{self.collection_name}'.")

    def similarity_search(self, query: str, k: int = 6) -> list[Document]:
        """
        Perform a similarity search against the vector store.
        Returns a list of langchain.schema.Document.
        """
        return self.vectorstore.similarity_search(query, k=k)
