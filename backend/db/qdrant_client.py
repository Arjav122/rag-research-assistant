from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from backend.utils.config import get_settings


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(vector_size: int = 3072) -> None:
    settings = get_settings()
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
