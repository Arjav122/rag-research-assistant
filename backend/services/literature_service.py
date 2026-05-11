"""Literature review: hybrid retrieval → grounded structured synthesis with citations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from openai import OpenAI

from backend.prompts.templates import LITERATURE_REVIEW_PROMPT
from backend.retrieval.context_builder import build_context, trim_neighbor_redundancy
from backend.retrieval.hybrid import hybrid_retrieve
from backend.retrieval.paper_key import normalize_paper_key
from backend.utils.config import get_settings

logger = logging.getLogger(__name__)


def _build_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-paper citations matching the numbered Context blocks the prompt references."""
    citations: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for idx, chunk in enumerate(chunks or [], start=1):
        meta = chunk.get("metadata") or {}
        key = normalize_paper_key(meta)
        if not key or key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "n": idx,
                "title": meta.get("title"),
                "source": meta.get("source"),
                "paper_id": meta.get("paper_id"),
                "arxiv_id": meta.get("arxiv_id"),
                "year": meta.get("year"),
                "authors": meta.get("authors") or [],
            }
        )
    return citations


def generate_literature_review(topic: str, max_papers: int) -> dict:
    chunks = hybrid_retrieve(query=topic, top_k=max_papers)
    if not chunks:
        return {
            "topic": topic,
            "review": (
                "No relevant passages were retrieved for this topic. "
                "Try broader phrasing, expand abbreviations (e.g. 'RAG' → 'retrieval-augmented generation'), "
                "or confirm the corpus has been indexed."
            ),
            "sources": 0,
            "citations": [],
            "papers": 0,
            "low_confidence": True,
        }

    chunks = trim_neighbor_redundancy(chunks)
    context = build_context(chunks)
    citations = _build_citations(chunks)

    settings = get_settings()
    prompt = LITERATURE_REVIEW_PROMPT.format(topic=topic, context=context)

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous research assistant. Never invent citations or papers "
                    "outside the provided Context. After every claim include inline [n] citation "
                    "markers that map to the numbered Context blocks."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    review_text = response.choices[0].message.content or ""

    # Confidence signal — if the strongest reranked chunk was poor, flag the review as
    # weakly grounded so the UI can surface a warning.
    max_score = float("-inf")
    for c in chunks:
        s = c.get("rerank_score", c.get("score", 0.0))
        if s is not None and float(s) > max_score:
            max_score = float(s)
    low_confidence = (
        settings.retrieval_use_confidence_guardrail
        and max_score < settings.retrieval_low_confidence_threshold
    )

    return {
        "topic": topic,
        "review": review_text,
        "sources": len(chunks),
        "papers": len(citations),
        "citations": citations,
        "low_confidence": bool(low_confidence),
    }
