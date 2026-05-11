"""Corpus-wide BM25 index for true parallel sparse retrieval.

Built lazily on first use, refreshed periodically (TTL). Thread-safe via a build lock so
concurrent requests don't all rebuild simultaneously. Reads chunks from Qdrant via scroll
and keeps everything in-process (~30k chunks fit comfortably in memory).

Important design notes:
- This module is opt-in via `settings.retrieval_use_corpus_bm25`. If anything misbehaves,
  flip the flag in `.env` and the system falls back to the original "BM25 over dense
  candidates" behavior — no other code path needs to change.
- We tokenize the same way for build and query (lowercase + simple word split) to keep
  rank_bm25's bag-of-words assumption consistent.
- Filters (e.g. `restrict_to_paper_id`) are applied at query time *after* BM25 scoring, by
  intersecting the top-N BM25 ids against a filter-aware Qdrant lookup. This avoids
  rebuilding per-filter indices.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from backend.db.qdrant_client import get_qdrant_client
from backend.utils.config import get_settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class _CorpusIndex:
    bm25: Optional[BM25Okapi] = None
    ids: List[str] = field(default_factory=list)
    payloads: Dict[str, dict] = field(default_factory=dict)
    built_at: float = 0.0
    chunk_count: int = 0


_index = _CorpusIndex()
_build_lock = threading.Lock()


def _scroll_all_chunks(max_chunks: int) -> Tuple[List[str], List[List[str]], Dict[str, dict]]:
    """Scroll the Qdrant collection and return ids, tokenized texts, and payloads."""
    settings = get_settings()
    client = get_qdrant_client()

    ids: List[str] = []
    tokenized: List[List[str]] = []
    payloads: Dict[str, dict] = {}

    next_offset = None
    page = 0
    page_size = 512
    while True:
        try:
            records, next_offset = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=page_size,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )
        except Exception:
            logger.exception("BM25 corpus scroll failed at page=%s", page)
            break

        if not records:
            break

        for r in records:
            sid = str(r.id)
            payload = r.payload or {}
            text = payload.get("text") or ""
            ids.append(sid)
            tokenized.append(_tokenize(text))
            payloads[sid] = payload
            if len(ids) >= max_chunks:
                logger.warning("BM25 corpus reached max_chunks=%s; truncating", max_chunks)
                return ids, tokenized, payloads

        page += 1
        if next_offset is None:
            break

    return ids, tokenized, payloads


def _build_index_locked() -> None:
    """Caller must hold `_build_lock`."""
    settings = get_settings()
    t0 = time.monotonic()
    ids, tokenized, payloads = _scroll_all_chunks(settings.retrieval_corpus_bm25_max_chunks)
    if not ids:
        logger.warning("BM25 corpus build skipped: no chunks scrolled from Qdrant")
        return
    bm25 = BM25Okapi(tokenized)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info("Built corpus BM25 index: chunks=%s in %sms", len(ids), elapsed_ms)
    _index.bm25 = bm25
    _index.ids = ids
    _index.payloads = payloads
    _index.built_at = time.monotonic()
    _index.chunk_count = len(ids)


def _index_is_fresh() -> bool:
    settings = get_settings()
    if _index.bm25 is None or not _index.ids:
        return False
    age = time.monotonic() - _index.built_at
    return age < settings.retrieval_corpus_bm25_refresh_seconds


def ensure_corpus_index() -> bool:
    """Build or refresh the index if stale. Returns True if a usable index is available."""
    if _index_is_fresh():
        return True
    with _build_lock:
        if _index_is_fresh():  # double-checked
            return True
        try:
            _build_index_locked()
        except Exception:
            logger.exception("BM25 corpus index build failed")
            return False
    return _index.bm25 is not None and bool(_index.ids)


def corpus_bm25_search(query: str, top_k: int) -> List[Dict]:
    """Return top-K chunks for `query` from the in-memory corpus BM25 index.

    Each item: {id, text, score, metadata}. Returns [] if index isn't usable.
    """
    if not query or top_k <= 0:
        return []
    if not ensure_corpus_index():
        return []

    bm25 = _index.bm25
    if bm25 is None:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    if scores.size == 0:
        return []

    n = min(top_k, scores.shape[0])
    top_idx = np.argpartition(-scores, n - 1)[:n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    out: List[Dict] = []
    for i in top_idx:
        if scores[i] <= 0:
            continue
        sid = _index.ids[i]
        payload = _index.payloads.get(sid) or {}
        out.append(
            {
                "id": sid,
                "text": payload.get("text", ""),
                "score": float(scores[i]),
                "metadata": payload,
            }
        )
    return out


def corpus_index_stats() -> Dict:
    return {
        "ready": _index.bm25 is not None,
        "chunks": _index.chunk_count,
        "age_seconds": int(time.monotonic() - _index.built_at) if _index.built_at else None,
    }
