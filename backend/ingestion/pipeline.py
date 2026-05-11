"""Orchestrates arXiv ingestion: download, chunk, embed, upsert with logging and per-paper error isolation."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List

from qdrant_client.models import PointStruct

from backend.db.qdrant_client import get_qdrant_client, ensure_collection
from backend.db.qdrant_indexed_papers import load_indexed_paper_ids
from backend.ingestion.arxiv_client import fetch_arxiv_papers
from backend.ingestion.chunker import semantic_chunk_text, summarize_chunk_batch
from backend.ingestion.embedder import embed_texts
from backend.ingestion.pdf_processor import download_pdf, extract_text_from_pdf
from backend.ingestion.verification import ingestion_verification_report
from backend.utils.config import get_settings

logger = logging.getLogger(__name__)


def _stable_point_id(arxiv_id: str, chunk_id: int) -> str:
    """Use the version-stripped arxiv id so re-ingesting v2 upserts over v1 chunks."""
    import re
    base = re.sub(r"v\d+$", "", arxiv_id or "")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://arxiv.org/abs/{base or arxiv_id}#chunk-{chunk_id}"))


async def ingest_arxiv_pipeline(
    topics: List[str],
    max_results: int,
    *,
    skip_if_indexed: bool = False,
    arxiv_start_offset: int = 0,
    arxiv_sort_descending: bool = True,
    arxiv_raw_query: str | None = None,
    arxiv_fallback_raw_query: str | None = None,
    arxiv_inter_request_delay_seconds: float | None = None,
) -> Dict[str, Any]:
    settings = get_settings()
    client = get_qdrant_client()
    ensure_collection()

    logger.info(
        "Starting arXiv ingestion: topics=%s max_results=%s skip_if_indexed=%s "
        "arxiv_start_offset=%s arxiv_sort_descending=%s raw_query=%s",
        topics,
        max_results,
        skip_if_indexed,
        arxiv_start_offset,
        arxiv_sort_descending,
        bool(arxiv_raw_query),
    )

    fetch_meta: dict[str, Any] = {}
    papers = await fetch_arxiv_papers(
        topics=topics,
        max_results=max_results,
        start_offset=arxiv_start_offset,
        sort_descending=arxiv_sort_descending,
        raw_query=arxiv_raw_query,
        fallback_raw_query=arxiv_fallback_raw_query,
        fetch_report=fetch_meta,
        inter_request_delay_seconds=arxiv_inter_request_delay_seconds,
    )

    logger.info(
        "Ingestion arXiv fetch summary: original_query=%r relaxed_query=%r est_total=%s "
        "effective_start=%s fetched=%s",
        fetch_meta.get("original_query"),
        fetch_meta.get("relaxed_query"),
        fetch_meta.get("estimated_total_results"),
        fetch_meta.get("effective_start_offset"),
        fetch_meta.get("papers_fetched"),
    )

    indexed_ids: set[str] | None = None
    if skip_if_indexed:
        indexed_ids = load_indexed_paper_ids(client, settings.qdrant_collection)
        logger.info("Skip-if-indexed: %s paper_id(s) already in Qdrant", len(indexed_ids))

    stats: Dict[str, Any] = {
        "papers_fetched": len(papers),
        "papers_indexed_ok": 0,
        "papers_skipped_already": 0,
        "papers_skipped": [],
        "chunks_indexed": 0,
        "errors": [],
        "chunk_stats_per_paper": [],
        "arxiv_fetch": fetch_meta,
    }

    upsert_batch = max(1, settings.qdrant_upsert_batch_size)

    for paper in papers:
        arxiv_id = paper.get("arxiv_id") or ""
        if not arxiv_id:
            stats["papers_skipped"].append({"reason": "missing_arxiv_id", "paper": paper})
            continue

        paper_id = str(paper.get("paper_id") or f"arxiv:{arxiv_id}")
        if indexed_ids is not None and paper_id in indexed_ids:
            stats["papers_skipped_already"] += 1
            logger.info("Skip already indexed: %s", paper_id)
            continue

        try:
            pdf_url = paper.get("pdf_url") or ""
            if not pdf_url:
                stats["papers_skipped"].append({"arxiv_id": arxiv_id, "reason": "no_pdf_url"})
                continue

            pdf_path = await download_pdf(pdf_url, "storage/papers", arxiv_id.replace("/", "_"))
            text = extract_text_from_pdf(pdf_path)
            if len(text) < settings.ingest_min_document_chars:
                stats["papers_skipped"].append(
                    {
                        "arxiv_id": arxiv_id,
                        "reason": "text_too_short",
                        "chars": len(text),
                        "min_required": settings.ingest_min_document_chars,
                    }
                )
                logger.warning("Skip %s: extracted text too short (%s chars)", arxiv_id, len(text))
                continue

            topic_label = paper.get("primary_category") or "AI"
            metadata = {
                "paper_id": paper.get("paper_id", f"arxiv:{arxiv_id}"),
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "authors": paper.get("authors") or [],
                "year": paper.get("year"),
                "topic": topic_label,
                "arxiv_categories": paper.get("arxiv_categories") or [],
                "primary_category": paper.get("primary_category"),
                "published": paper.get("published", ""),
                "source": "arxiv",
            }

            chunks = semantic_chunk_text(text=text, metadata=metadata)
            if not chunks:
                stats["papers_skipped"].append({"arxiv_id": arxiv_id, "reason": "no_chunks_after_filter"})
                continue

            # Inject the arXiv-supplied abstract as chunk_id=0 with section="abstract".
            # This is the highest-signal text per paper and is *not* always present in the PDF text.
            abstract_text = (paper.get("summary") or "").strip()
            if abstract_text and len(abstract_text) >= settings.ingest_min_chunk_chars:
                # Renumber existing chunks to make room for the abstract at index 0.
                for c in chunks:
                    c["metadata"]["chunk_id"] = c["metadata"]["chunk_id"] + 1
                chunks.insert(
                    0,
                    {
                        "text": abstract_text,
                        "metadata": {
                            **metadata,
                            "section": "abstract",
                            "chunk_id": 0,
                            "chunk_char_len": len(abstract_text),
                            "is_abstract": True,
                        },
                    },
                )

            batch_summary = summarize_chunk_batch(chunks)
            stats["chunk_stats_per_paper"].append({"arxiv_id": arxiv_id, **batch_summary})

            texts = [c["text"] for c in chunks]
            vectors = embed_texts(texts)

            points: List[PointStruct] = []
            for idx, vector in enumerate(vectors):
                meta = chunks[idx]["metadata"]
                chunk_id = int(meta["chunk_id"])
                pid = _stable_point_id(arxiv_id, chunk_id)
                points.append(
                    PointStruct(
                        id=pid,
                        vector=vector,
                        payload={
                            **meta,
                            "text": texts[idx],
                        },
                    )
                )

            for i in range(0, len(points), upsert_batch):
                batch = points[i : i + upsert_batch]
                client.upsert(collection_name=settings.qdrant_collection, points=batch)

            stats["papers_indexed_ok"] += 1
            stats["chunks_indexed"] += len(points)
            if indexed_ids is not None:
                indexed_ids.add(paper_id)
            logger.info(
                "Indexed arxiv:%s chunks=%s (chars_mean=%s)",
                arxiv_id,
                len(points),
                batch_summary.get("mean_chars"),
            )

        except Exception as exc:
            logger.exception("Ingestion failed for arxiv:%s", arxiv_id)
            stats["errors"].append(
                {
                    "arxiv_id": arxiv_id,
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )

    try:
        stats["qdrant_verification"] = ingestion_verification_report(client)
    except Exception:
        logger.exception("Post-ingestion Qdrant verification failed")
        stats["qdrant_verification"] = {"error": "verification_failed"}

    logger.info(
        "Ingestion complete: newly_indexed=%s duplicates_skipped=%s chunks=%s "
        "errors=%s other_skipped=%s (fetch_returned=%s)",
        stats["papers_indexed_ok"],
        stats["papers_skipped_already"],
        stats["chunks_indexed"],
        len(stats["errors"]),
        len(stats["papers_skipped"]),
        len(papers),
    )
    return stats
