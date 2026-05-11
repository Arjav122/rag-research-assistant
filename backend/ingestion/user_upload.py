"""Index a user-uploaded PDF into Qdrant (same chunk/embed path as arXiv)."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client.models import PointStruct

from backend.db.qdrant_client import get_qdrant_client, ensure_collection
from backend.ingestion.chunker import semantic_chunk_text, summarize_chunk_batch
from backend.ingestion.embedder import embed_texts
from backend.ingestion.pdf_processor import extract_text_from_pdf
from backend.utils.config import get_settings

logger = logging.getLogger(__name__)


def _upload_point_id(upload_key: str, chunk_id: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"user-upload:{upload_key}#chunk-{chunk_id}"))


def ingest_user_pdf_bytes(file_content: bytes, original_filename: str) -> Dict[str, Any]:
    settings = get_settings()
    client = get_qdrant_client()
    ensure_collection()

    upload_key = str(uuid.uuid4())
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in original_filename)[:160]
    out_dir = Path("storage/uploads") / upload_key
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / (safe_name if safe_name.lower().endswith(".pdf") else f"{safe_name}.pdf")
    pdf_path.write_bytes(file_content)

    text = extract_text_from_pdf(pdf_path)
    if len(text) < settings.ingest_min_document_chars:
        return {
            "success": False,
            "error": "extracted_text_too_short",
            "chars": len(text),
            "min_required": settings.ingest_min_document_chars,
        }

    title_guess = Path(original_filename).stem.replace("_", " ")
    metadata = {
        "paper_id": f"user:{upload_key}",
        "arxiv_id": upload_key,
        "title": title_guess,
        "authors": [],
        "year": None,
        "topic": "user_upload",
        "arxiv_categories": [],
        "primary_category": None,
        "published": "",
        "source": "user_upload",
        "original_filename": original_filename,
    }

    chunks = semantic_chunk_text(text=text, metadata=metadata)
    if not chunks:
        return {"success": False, "error": "no_chunks_after_filter"}

    # If section detection found an abstract chunk, promote it to chunk_id=0 with is_abstract=True.
    abstract_chunks = [c for c in chunks if (c["metadata"].get("section") == "abstract")]
    if abstract_chunks:
        abstract_chunks[0]["metadata"]["is_abstract"] = True

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    points: List[PointStruct] = []
    for idx, vector in enumerate(vectors):
        meta = chunks[idx]["metadata"]
        chunk_id = int(meta["chunk_id"])
        pid = _upload_point_id(upload_key, chunk_id)
        points.append(
            PointStruct(
                id=pid,
                vector=vector,
                payload={**meta, "text": texts[idx]},
            )
        )

    upsert_batch = max(1, settings.qdrant_upsert_batch_size)
    for i in range(0, len(points), upsert_batch):
        client.upsert(collection_name=settings.qdrant_collection, points=points[i : i + upsert_batch])

    summary = summarize_chunk_batch(chunks)
    logger.info("User upload indexed upload_key=%s chunks=%s", upload_key, len(points))

    return {
        "success": True,
        "upload_id": upload_key,
        "paper_id": metadata["paper_id"],
        "title": title_guess,
        "chunks_indexed": len(points),
        "chunk_stats": summary,
    }
