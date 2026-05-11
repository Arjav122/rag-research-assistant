"""Hybrid retrieval: dense + true-parallel BM25 + (optional) HyDE + section-aware boost.

Public API is unchanged: `hybrid_retrieve(query, top_k, qdrant_filter)` returns a list of
chunks with `rerank_score` set on each. New behaviors are gated by config flags so the
pre-existing pipeline can be restored by flipping flags in `.env`.

Pipeline (Tier 1 enabled):
  query
    → query_rewriter (static abbrev expansion + LLM follow-up self-contain + optional HyDE)
    → dense retrieval (expanded query)  ──┐
    → dense retrieval (HyDE doc)  ────────┤  union
    → corpus-wide BM25 retrieval ─────────┤
    → fetch any BM25-only chunks from Qdrant by id (so we can rerank them)
    → RRF on three rankings (dense_expanded, dense_hyde, bm25_corpus)
    → metadata + section-intent boost
    → per-paper prefetch dedupe
    → cross-encoder rerank (BAAI/bge-reranker-large)
    → diversify by paper
    → top-k
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

from openai import OpenAI

from backend.db.qdrant_client import get_qdrant_client
from backend.retrieval.bm25 import bm25_scores_all
from backend.retrieval.bm25_corpus import corpus_bm25_search
from backend.retrieval.fusion import (
    diversify_by_paper,
    metadata_relevance_boost,
    reciprocal_rank_fusion,
)
from backend.retrieval.paper_key import chunk_paper_key
from backend.retrieval.query_rewriter import (
    RewriteResult,
    detect_comparison_entities,
    prepare_query,
)
from backend.retrieval.reranker import rerank
from backend.retrieval.section_intent import classify_section_intent, neighbor_sections
from backend.utils.config import get_settings

logger = logging.getLogger(__name__)
_EMBED_CACHE_LOCK = threading.Lock()
_EMBED_CACHE: Dict[str, List[float]] = {}
_EMBED_CACHE_MAX = 256


def _embed_query(query: str) -> List[float]:
    q = (query or "").strip()
    if not q:
        return []
    with _EMBED_CACHE_LOCK:
        cached = _EMBED_CACHE.get(q)
    if cached is not None:
        return cached

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(model=settings.openai_embedding_model, input=q)
    vec = response.data[0].embedding
    with _EMBED_CACHE_LOCK:
        if q not in _EMBED_CACHE:
            if len(_EMBED_CACHE) >= _EMBED_CACHE_MAX:
                _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))
            _EMBED_CACHE[q] = vec
    return vec


def _prefetch_limit(top_k: int, multiplier: int) -> int:
    return min(240, max(top_k * multiplier, top_k * 4))


def _dedupe_prefetch_pool(
    scored_docs: List[Dict],
    max_per_paper: int,
    pool_cap: int,
) -> List[Dict]:
    """Cap chunks per paper using the version-normalized key.

    Using the raw `paper_id` here let `arxiv:xxxxv1` and `arxiv:xxxx` (same paper, two
    ingestion eras) bypass the per-paper cap, which is the dominant source of the
    "duplicate papers in top-k" symptom.
    """
    by_paper: Dict[str, List[Dict]] = {}
    for doc in scored_docs:
        pk = chunk_paper_key(doc) or str(doc.get("id"))
        by_paper.setdefault(pk, []).append(doc)

    merged: List[Dict] = []
    for _, items in by_paper.items():
        items_sorted = sorted(items, key=lambda x: float(x.get("fusion_score", 0)), reverse=True)
        merged.extend(items_sorted[:max_per_paper])

    merged.sort(key=lambda x: float(x.get("fusion_score", 0)), reverse=True)
    return merged[:pool_cap]


def _vector_search(query_text: str, prefetch: int, qdrant_filter):
    settings = get_settings()
    client = get_qdrant_client()
    vector = _embed_query(query_text)
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=prefetch,
        query_filter=qdrant_filter,
    )
    return list(response.points)


def _hits_to_documents(hits) -> List[Dict]:
    documents: List[Dict] = []
    for h in hits:
        payload = h.payload or {}
        documents.append(
            {
                "id": str(h.id),
                "score": float(h.score) if hasattr(h, "score") and h.score is not None else 0.0,
                "text": payload.get("text", ""),
                "metadata": payload,
            }
        )
    return documents


def _bm25_corpus_with_filter(
    query_text: str,
    top_k: int,
    qdrant_filter,
) -> List[Dict]:
    """Run corpus-wide BM25, then apply the same Qdrant filter post-hoc.

    For unfiltered queries this is just `corpus_bm25_search`. When a filter is set (e.g.
    `restrict_to_paper_id`), we apply it client-side via the cached payloads — avoids a
    Qdrant round-trip and keeps BM25 fully parallel to the dense path.
    """
    raw = corpus_bm25_search(query_text, top_k=top_k)
    if not qdrant_filter or not raw:
        return raw

    # Decode the Qdrant filter into simple field predicates we can apply locally.
    # We support `must` field-equality filters (paper_id, source) — our app uses only those.
    must = getattr(qdrant_filter, "must", None) or []
    field_eq: Dict[str, str] = {}
    for cond in must:
        field = getattr(cond, "key", None)
        value = None
        m = getattr(cond, "match", None)
        if m is not None:
            value = getattr(m, "value", None)
        if field and value is not None:
            field_eq[field] = value
    if not field_eq:
        return raw

    out: List[Dict] = []
    for doc in raw:
        meta = doc.get("metadata") or {}
        if all(meta.get(k) == v for k, v in field_eq.items()):
            out.append(doc)
    return out


def _fetch_payloads_by_ids(ids: List[str]) -> List[Dict]:
    """Look up Qdrant payloads for ids that came only from the BM25 path."""
    if not ids:
        return []
    settings = get_settings()
    client = get_qdrant_client()
    try:
        records = client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        logger.exception("Qdrant retrieve(ids=...) failed for BM25-only candidates")
        return []
    out: List[Dict] = []
    for r in records:
        payload = r.payload or {}
        out.append(
            {
                "id": str(r.id),
                "score": 0.0,
                "text": payload.get("text", ""),
                "metadata": payload,
            }
        )
    return out


def _section_intent_boost(
    metadata: Dict,
    target: Optional[str],
    boost: float,
) -> float:
    if not target or boost <= 0:
        return 0.0
    section = (metadata.get("section") or "").lower()
    if section == target:
        return boost
    if section in neighbor_sections(target):
        return boost * 0.4
    return 0.0


def hybrid_retrieve(
    query: str,
    top_k: int = 10,
    *,
    qdrant_filter: Optional[object] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict]:
    """Hybrid retrieval with optional Tier 1 intelligence layers.

    `history` is forwarded to the query rewriter so follow-up turns become self-contained.
    All Tier 1 features are individually flag-gated (see `backend.utils.config.Settings`).
    """
    settings = get_settings()
    t_all = time.monotonic()

    # ----- 1. Query understanding (cheap, optional) -----
    t_rewrite = time.monotonic()
    if settings.retrieval_use_query_rewrite:
        rewrite = prepare_query(query, history=history)
    else:
        # Even with full rewrite disabled we still want cheap entity detection so the
        # multi-entity sub-retrieval kicks in for "X vs Y" queries.
        rewrite = RewriteResult(
            expanded_query=query,
            hyde_text=None,
            raw_query=query,
            used_llm=False,
            entities=detect_comparison_entities(query),
        )
    expanded_query = rewrite.expanded_query or query
    hyde_text = rewrite.hyde_text
    entities = rewrite.entities or []
    rewrite_ms = int((time.monotonic() - t_rewrite) * 1000)

    section_target = classify_section_intent(query) if settings.retrieval_use_section_intent else None
    if section_target:
        logger.debug("Section intent for query=%r → %s", query, section_target)

    prefetch = _prefetch_limit(top_k, settings.retrieval_prefetch_multiplier)

    # ----- 2. Dense retrieval (expanded query) -----
    t_dense = time.monotonic()
    dense_hits = _vector_search(expanded_query, prefetch, qdrant_filter)
    if not dense_hits and expanded_query != query:
        # If the LLM-rewritten query was too aggressive, fall back to the raw query.
        dense_hits = _vector_search(query, prefetch, qdrant_filter)
    dense_ms = int((time.monotonic() - t_dense) * 1000)

    # ----- 3. (optional) Dense retrieval over HyDE document -----
    t_hyde = time.monotonic()
    hyde_hits = []
    if hyde_text:
        try:
            hyde_hits = _vector_search(hyde_text, prefetch, qdrant_filter)
        except Exception:
            logger.exception("HyDE vector search failed; ignoring")
    hyde_ms = int((time.monotonic() - t_hyde) * 1000)

    # ----- 3b. (optional) Per-entity dense retrieval for "X vs Y"-style queries -----
    # Each entity gets its own embedding probe so neither side is diluted in the main
    # query vector. Smaller prefetch keeps cost bounded; RRF naturally weights this
    # alongside the other rankings.
    entity_hits_per: List[List] = []
    t_entity = time.monotonic()
    if entities and len(entities) >= 2:
        entity_prefetch = max(top_k * 3, 24)
        for ent in entities:
            try:
                eh = _vector_search(ent, entity_prefetch, qdrant_filter)
            except Exception:
                logger.exception("Per-entity vector search failed for entity=%r; skipping", ent)
                eh = []
            entity_hits_per.append(eh)
    entity_ms = int((time.monotonic() - t_entity) * 1000)

    # ----- 4. (optional) Corpus-wide BM25 -----
    t_bm25 = time.monotonic()
    bm25_corpus_hits: List[Dict] = []
    if settings.retrieval_use_corpus_bm25:
        try:
            bm25_corpus_hits = _bm25_corpus_with_filter(
                expanded_query,
                top_k=settings.retrieval_corpus_bm25_top_k,
                qdrant_filter=qdrant_filter,
            )
        except Exception:
            logger.exception("Corpus BM25 retrieval failed; continuing without it")
    bm25_ms = int((time.monotonic() - t_bm25) * 1000)

    # ----- 5. Build the unified candidate pool -----
    documents: Dict[str, Dict] = {}
    dense_ids_ordered: List[str] = []
    hyde_ids_ordered: List[str] = []
    bm25_ids_ordered: List[str] = []
    entity_ids_ordered_per: List[List[str]] = []

    for h in dense_hits:
        sid = str(h.id)
        dense_ids_ordered.append(sid)
        if sid not in documents:
            documents[sid] = _hits_to_documents([h])[0]
    for h in hyde_hits:
        sid = str(h.id)
        hyde_ids_ordered.append(sid)
        if sid not in documents:
            documents[sid] = _hits_to_documents([h])[0]
    for d in bm25_corpus_hits:
        sid = str(d["id"])
        bm25_ids_ordered.append(sid)
        if sid not in documents:
            documents[sid] = d
    for entity_hits in entity_hits_per:
        ordered: List[str] = []
        for h in entity_hits:
            sid = str(h.id)
            ordered.append(sid)
            if sid not in documents:
                documents[sid] = _hits_to_documents([h])[0]
        entity_ids_ordered_per.append(ordered)

    if not documents:
        return []

    # Some BM25-only ids may not have been fetched yet via Qdrant; that's fine — the
    # corpus BM25 cache already includes payload+text, so we have what we need.

    # ----- 6. Local BM25 over the dense candidate set (legacy signal, still useful) -----
    t_fusion = time.monotonic()
    docs_list = list(documents.values())
    bm25_local_dense = bm25_scores_all(query, docs_list)
    bm25_local_order_idx = sorted(range(len(docs_list)), key=lambda i: bm25_local_dense[i], reverse=True)
    bm25_local_ids_ordered = [docs_list[i]["id"] for i in bm25_local_order_idx]

    rankings = [
        r
        for r in [dense_ids_ordered, bm25_local_ids_ordered, bm25_ids_ordered, hyde_ids_ordered]
        if r
    ]
    # Each entity's ranking is added independently so a paper supporting only one entity
    # still gets a fair RRF boost (instead of being averaged into nothing).
    rankings.extend([r for r in entity_ids_ordered_per if r])
    rrf_scores = reciprocal_rank_fusion(rankings, k=settings.retrieval_rrf_k)

    # ----- 7. Metadata + section-intent boost -----
    section_boost = settings.retrieval_section_intent_boost
    for doc in docs_list:
        sid = str(doc["id"])
        meta = doc.get("metadata") or {}
        boost = metadata_relevance_boost(query, meta, weight=settings.retrieval_metadata_boost_cap)
        section_b = _section_intent_boost(meta, section_target, section_boost)
        doc["fusion_score"] = rrf_scores.get(sid, 0.0) + boost + section_b
        doc["section_intent"] = section_target

    scored_sorted = sorted(docs_list, key=lambda x: float(x["fusion_score"]), reverse=True)
    prefetch_pool = _dedupe_prefetch_pool(
        scored_sorted,
        max_per_paper=settings.retrieval_max_chunks_per_paper_prefetch,
        pool_cap=settings.retrieval_rerank_pool_cap,
    )
    fusion_ms = int((time.monotonic() - t_fusion) * 1000)

    rerank_take = min(len(prefetch_pool), max(top_k * 3, top_k))
    t_rerank = time.monotonic()
    reranked = rerank(query=query, candidates=prefetch_pool, top_k=rerank_take)
    rerank_ms = int((time.monotonic() - t_rerank) * 1000)

    final = diversify_by_paper(
        reranked,
        top_k=top_k,
        max_per_paper=settings.retrieval_final_max_per_paper,
    )
    total_ms = int((time.monotonic() - t_all) * 1000)
    logger.info(
        "Hybrid timing: total_ms=%s rewrite_ms=%s dense_ms=%s hyde_ms=%s entity_ms=%s bm25_ms=%s fusion_ms=%s rerank_ms=%s candidates=%s",
        total_ms,
        rewrite_ms,
        dense_ms,
        hyde_ms,
        entity_ms,
        bm25_ms,
        fusion_ms,
        rerank_ms,
        len(prefetch_pool),
    )
    return final
