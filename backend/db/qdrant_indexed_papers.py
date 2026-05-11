"""Load distinct `paper_id` values already stored in Qdrant (for ingestion skip-if-exists)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from qdrant_client import QdrantClient


def summarize_corpus(client: QdrantClient, collection_name: str) -> dict[str, Any]:
    """Unique papers, point count, and topic (primary category) distribution from chunk payloads."""
    paper_ids: set[str] = set()
    topic_counts: Counter[str] = Counter()
    offset = None
    scrolled = 0
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=1024,
            offset=offset,
            with_payload=["paper_id", "topic", "primary_category"],
            with_vectors=False,
        )
        for r in records:
            scrolled += 1
            pl = r.payload or {}
            pid = pl.get("paper_id")
            if pid:
                paper_ids.add(str(pid))
            t = pl.get("topic") or pl.get("primary_category") or ""
            if t:
                topic_counts[str(t)] += 1
        if offset is None:
            break
    coll = client.get_collection(collection_name)
    return {
        "unique_papers": len(paper_ids),
        "points_count": coll.points_count,
        "chunks_scrolled": scrolled,
        "topic_distribution": topic_counts.most_common(25),
    }


def load_indexed_paper_ids(client: QdrantClient, collection_name: str) -> set[str]:
    """Scroll payloads (paper_id only). Safe for large collections; one scroll batch at a time."""
    out: set[str] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=1024,
            offset=offset,
            with_payload=["paper_id"],
            with_vectors=False,
        )
        for r in records:
            pl = r.payload or {}
            pid = pl.get("paper_id")
            if pid:
                out.add(str(pid))
        if offset is None:
            break
    return out
