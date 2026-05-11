"""RRF and lightweight metadata boosts — refinement layer on top of vector + BM25."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from backend.retrieval.paper_key import chunk_paper_key


def reciprocal_rank_fusion(
    ranked_ids_lists: List[List[str]],
    k: int = 60,
) -> Dict[str, float]:
    """Standard RRF: score(id) = sum_i 1/(k + rank_i)."""
    scores: Dict[str, float] = {}
    for ranked_ids in ranked_ids_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def metadata_relevance_boost(query: str, metadata: Dict[str, Any], weight: float = 0.08) -> float:
    """Small boost for title/category/topic overlap, plus a section-aware nudge.

    The boost is intentionally capped so vector + BM25 + reranker remain the dominant signals.
    Section-aware nudge: abstract chunks get a small lift since they summarize the whole paper.
    """
    if not query.strip():
        return 0.0
    q_tokens = {t.lower() for t in query.split() if len(t) > 2}
    if not q_tokens:
        return 0.0

    title = (metadata.get("title") or "").lower()
    cats = " ".join(metadata.get("arxiv_categories") or []).lower()
    topic = (metadata.get("topic") or "").lower()
    blob = f"{title} {cats} {topic}"

    overlap = sum(1 for t in q_tokens if t in blob)
    score = min(weight, overlap * (weight / 4.0))

    # Section-aware: abstract chunks summarize the whole paper, so give them a small extra lift.
    section = (metadata.get("section") or "").lower()
    if section == "abstract" or metadata.get("is_abstract"):
        score += weight * 0.5
    elif section in ("introduction", "method", "results", "conclusion"):
        score += weight * 0.15

    return min(weight * 1.5, score)


def normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(hi, lo):
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def diversify_by_paper(
    chunks: List[Dict[str, Any]],
    top_k: int,
    max_per_paper: int = 2,
) -> List[Dict[str, Any]]:
    """Prefer multiple papers in top-k using a **breadth-first** pass.

    The legacy greedy "take up to max_per_paper immediately" walk meant the reranker
    could place two chunks from paper A before paper B's best chunk — yielding visible
    duplicate-paper leakage at the top.

    Here we first take **at most one chunk per paper** (in global rerank order), then a
    second pass grants additional chunks up to `max_per_paper`. No third pass —
    avoids stuffing the tail with triples from one paper unless the caller increases
    `max_per_paper` deliberately.
    """
    if top_k <= 0 or not chunks:
        return []

    out: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    counts: Dict[str, int] = {}

    def add(c: Dict[str, Any]) -> bool:
        cid = str(c.get("id") or "")
        if not cid:
            return False
        if cid in used_ids:
            return False
        pk = chunk_paper_key(c)
        if counts.get(pk, 0) >= max_per_paper:
            return False
        out.append(c)
        used_ids.add(cid)
        counts[pk] = counts.get(pk, 0) + 1
        return True

    # Pass 1: one chunk per paper, preserve reranked order across papers.
    for c in chunks:
        if len(out) >= top_k:
            return out
        pk = chunk_paper_key(c)
        if counts.get(pk, 0) == 0:
            add(c)

    # Pass 2: second slices for papers already represented.
    for c in chunks:
        if len(out) >= top_k:
            return out
        add(c)

    return out[:top_k]
