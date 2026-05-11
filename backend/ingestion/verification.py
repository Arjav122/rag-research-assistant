"""Post-ingestion checks against Qdrant (counts, optional sample scroll)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient

from backend.utils.config import get_settings

logger = logging.getLogger(__name__)

EXPECTED_DIM_LARGE = 3072


def collection_point_count(client: QdrantClient, collection_name: str | None = None) -> int:
    name = collection_name or get_settings().qdrant_collection
    result = client.count(collection_name=name, exact=True)
    return int(result.count)


def _vector_size_from_collection_info(info: Any) -> int | None:
    params = info.config.params.vectors
    if params is None:
        return None
    if isinstance(params, dict):
        if not params:
            return None
        first = next(iter(params.values()))
        return getattr(first, "size", None)
    return getattr(params, "size", None)


def verify_collection_vector_size(client: QdrantClient, expected_dim: int = EXPECTED_DIM_LARGE) -> Dict[str, Any]:
    settings = get_settings()
    info = client.get_collection(settings.qdrant_collection)
    size = _vector_size_from_collection_info(info)
    if size is None:
        return {"ok": False, "error": "could not read vector size", "vector_size": None}
    ok = size == expected_dim
    if not ok:
        logger.warning("Qdrant vector size %s != expected %s", size, expected_dim)
    return {"ok": ok, "vector_size": size, "expected": expected_dim}


def scroll_sample_payloads(
    client: QdrantClient,
    limit: int = 3,
    collection_name: str | None = None,
) -> List[Dict[str, Any]]:
    """Return small payload samples for manual retrieval sanity checks."""
    name = collection_name or get_settings().qdrant_collection
    records, _ = client.scroll(collection_name=name, limit=limit, with_payload=True, with_vectors=False)
    out: List[Dict[str, Any]] = []
    for r in records:
        p = r.payload or {}
        out.append(
            {
                "id": str(r.id),
                "title": p.get("title"),
                "arxiv_id": p.get("arxiv_id"),
                "chunk_id": p.get("chunk_id"),
                "text_preview": (p.get("text") or "")[:200],
            }
        )
    return out


def ingestion_verification_report(client: QdrantClient) -> Dict[str, Any]:
    settings = get_settings()
    count = collection_point_count(client, settings.qdrant_collection)
    vs = verify_collection_vector_size(client)
    samples = scroll_sample_payloads(client, limit=3) if count else []
    return {
        "collection": settings.qdrant_collection,
        "points_count": count,
        "vector_config": vs,
        "sample_payloads": samples,
    }
