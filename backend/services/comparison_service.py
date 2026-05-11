"""Paper comparison: input resolution → shared-anchor retrieval per paper → grounded synthesis.

Why this shape:
- The previous version blindly treated every input string as a Qdrant id, so any
  natural-language input ("LatentRAG vs Superintelligent Retrieval Agent") returned
  zero chunks. Now we resolve titles → canonical ids via hybrid retrieval before
  Qdrant lookup.
- The previous version scrolled chunks per paper independently (first/middle/last
  positional spread), which gave the LLM unaligned evidence and produced generic
  "summary then summary" output. Now we derive a shared anchor query from all paper
  titles and retrieve each paper's most-relevant-to-the-anchor chunks via filtered
  hybrid retrieval — so the LLM sees aligned cross-paper evidence on the same theme.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from backend.db.qdrant_client import get_qdrant_client
from backend.prompts.templates import COMPARISON_PROMPT
from backend.retrieval.hybrid import hybrid_retrieve
from backend.retrieval.paper_key import normalize_paper_key
from backend.retrieval.qdrant_filters import build_retrieval_filter
from backend.utils.config import get_settings

logger = logging.getLogger(__name__)


# Conservative caps so the comparison prompt stays under token limits.
MAX_CHUNKS_PER_PAPER = 6
MAX_CHARS_PER_CHUNK = 1200
_RESULTS_SECTION_HINTS = frozenset(
    {"results", "experiment", "experiments", "evaluation"}
)

# Strip these instruction-y wrappers from user input before treating it as a paper handle.
_INSTRUCTION_PREFIX_RE = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+)?(?:compare|contrast|diff|vs\.?|versus|between|and|or)\s*[:\-]?\s*",
    re.IGNORECASE,
)
_QUOTE_CHARS = "\"'`“”‘’"
_HANDLE_RE = re.compile(r"^(arxiv:|user:)", re.IGNORECASE)
_BARE_ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def _looks_like_handle(s: str) -> bool:
    s = s.strip()
    return bool(_HANDLE_RE.match(s) or _BARE_ARXIV_RE.match(s))


def _strip_input(raw: str) -> str:
    """Normalize a single input line: strip quotes, instruction prefixes, trailing punctuation."""
    s = (raw or "").strip()
    if not s:
        return ""
    s = s.strip(_QUOTE_CHARS).strip()
    s = _INSTRUCTION_PREFIX_RE.sub("", s).strip()
    s = s.strip(",;.").strip()
    return s


def _resolve_paper_input(raw: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Resolve a raw user input into a canonical paper_id, plus a resolution note.

    Returns:
      (canonical_paper_id_or_None, info_dict)
      info_dict carries fields the UI can show: input, resolved_from ("handle" |
      "title-search" | None), candidate_title, candidate_score.
    """
    cleaned = _strip_input(raw)
    info: Dict[str, Any] = {"input": raw, "cleaned_input": cleaned, "resolved_from": None}

    if not cleaned:
        info["error"] = "empty input"
        return None, info

    # Direct handle path — keep arxiv:/user: prefixes; promote bare arxiv ids.
    if _looks_like_handle(cleaned):
        if cleaned.lower().startswith(("arxiv:", "user:")):
            info["resolved_from"] = "handle"
            return cleaned, info
        info["resolved_from"] = "handle"
        return f"arxiv:{cleaned}", info

    # Title path — search the index, pick the paper most strongly matched.
    # Small top_k keeps resolution fast (full hybrid still runs, but fewer chunks to rerank/diversify).
    try:
        hits = hybrid_retrieve(query=cleaned, top_k=5)
    except Exception:
        logger.exception("hybrid_retrieve failed during paper input resolution")
        hits = []
    if not hits:
        info["error"] = "no matching paper in index"
        return None, info

    # Aggregate by normalized paper key, taking the best chunk per paper.
    by_paper: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        key = normalize_paper_key(h.get("metadata") or {})
        if not key:
            continue
        score = float(h.get("rerank_score", h.get("score", 0.0)) or 0.0)
        existing = by_paper.get(key)
        if existing is None or score > float(existing.get("rerank_score", 0.0) or 0.0):
            by_paper[key] = h

    if not by_paper:
        info["error"] = "no paper-keyed chunks"
        return None, info

    best = max(
        by_paper.values(),
        key=lambda h: float(h.get("rerank_score", h.get("score", 0.0)) or 0.0),
    )
    meta = best.get("metadata") or {}
    canonical = (meta.get("paper_id") or "").strip()
    if not canonical:
        # Fall back to constructing arxiv:<id> from arxiv_id; still better than nothing.
        arxiv = (meta.get("arxiv_id") or "").strip()
        canonical = f"arxiv:{arxiv}" if arxiv else ""

    if not canonical:
        info["error"] = "best match had no paper_id metadata"
        return None, info

    info["resolved_from"] = "title-search"
    info["candidate_title"] = meta.get("title")
    info["candidate_score"] = round(float(best.get("rerank_score", 0.0) or 0.0), 3)
    return canonical, info


def _shared_anchor_query(titles: List[str]) -> str:
    """Build a single retrieval query that captures the comparison theme across papers.

    We just glue the titles together with "key methods, results, and limitations" — this
    is intentionally lightweight (no extra LLM call). The reranker handles the rest.
    """
    cleaned = [t.strip() for t in titles if t and t.strip()]
    if not cleaned:
        return "key methods, results, and limitations"
    joined = "; ".join(cleaned[:4])
    return f"{joined} — key methods, datasets, results, and limitations"


def _section_priority(meta: Dict[str, Any]) -> int:
    s = (meta.get("section") or "").lower()
    if any(h in s for h in _RESULTS_SECTION_HINTS):
        return 2
    return 0


def _row_from_hybrid_chunk(c: Dict[str, Any]) -> Dict[str, Any]:
    meta = c.get("metadata") or {}
    return {
        "chunk_id": meta.get("chunk_id"),
        "title": meta.get("title"),
        "authors": meta.get("authors") or [],
        "year": meta.get("year"),
        "arxiv_id": meta.get("arxiv_id"),
        "paper_id": meta.get("paper_id"),
        "section": meta.get("section"),
        "text": (c.get("text") or "")[:MAX_CHARS_PER_CHUNK],
    }


def _probe_one_chunk(paper_id: str, anchor_query: str) -> List[Dict[str, Any]]:
    """Single hybrid call for title discovery — avoids paying the dual-query merge cost."""
    pid = (paper_id or "").strip()
    if not pid:
        return []
    qf = build_retrieval_filter(retrieval_scope="all", restrict_to_paper_id=pid)
    try:
        raw = hybrid_retrieve(query=anchor_query, top_k=1, qdrant_filter=qf)
    except Exception:
        logger.exception("hybrid probe failed for paper_id=%s", pid)
        return []
    if not raw:
        return []
    return [_row_from_hybrid_chunk(raw[0])]


def _fetch_anchored_chunks(
    paper_id: str,
    anchor_query: str,
    limit: int = MAX_CHUNKS_PER_PAPER,
) -> List[Dict[str, Any]]:
    """Single filtered hybrid call with anchor + results hints in one query.

    Previously this ran two full hybrid_retrievals per paper (anchor + results boost),
    which doubled latency (each hybrid = embeddings + rerank on CPU). One merged query
    preserves most recall for comparisons while cutting wall time roughly in half here.
    """
    pid = (paper_id or "").strip()
    if not pid:
        return []
    qf = build_retrieval_filter(retrieval_scope="all", restrict_to_paper_id=pid)
    take = min(limit + 2, 8)
    combined_query = (
        f"{anchor_query} quantitative experimental results benchmarks metrics "
        "evaluation accuracy performance tables ablation"
    )
    try:
        primary = hybrid_retrieve(query=combined_query, top_k=take, qdrant_filter=qf)
    except Exception:
        logger.exception("hybrid_retrieve failed for paper_id=%s", pid)
        return []

    ranked = sorted(
        primary,
        key=lambda c: (
            -_section_priority(c.get("metadata") or {}),
            -float(c.get("rerank_score", c.get("score", 0.0)) or 0.0),
        ),
    )
    ranked = ranked[:limit]
    return [_row_from_hybrid_chunk(c) for c in ranked]


def _scroll_fallback(paper_id: str, limit: int = MAX_CHUNKS_PER_PAPER) -> List[Dict[str, Any]]:
    """Fallback when filtered retrieval comes up empty (e.g. very short paper, anchor
    query missed). Direct Qdrant scroll, no scoring, document order."""
    settings = get_settings()
    client = get_qdrant_client()
    qf = build_retrieval_filter(retrieval_scope="all", restrict_to_paper_id=paper_id)
    try:
        records, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=qf,
            limit=64,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        logger.exception("Qdrant scroll fallback failed for paper_id=%s", paper_id)
        return []

    chunks: List[Dict[str, Any]] = []
    for r in records:
        payload = r.payload or {}
        chunks.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "title": payload.get("title"),
                "authors": payload.get("authors") or [],
                "year": payload.get("year"),
                "arxiv_id": payload.get("arxiv_id"),
                "paper_id": payload.get("paper_id"),
                "section": payload.get("section"),
                "text": (payload.get("text") or "")[:MAX_CHARS_PER_CHUNK],
            }
        )
    chunks.sort(key=lambda c: int(c.get("chunk_id") or 0))
    return chunks[:limit]


def _format_paper_block(label: str, chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return f"### {label}\n(No chunks were retrieved for this paper. It may not be indexed.)\n"
    head = chunks[0]
    title = head.get("title") or "Unknown title"
    authors = head.get("authors") or []
    year = head.get("year")
    arxiv_id = head.get("arxiv_id") or ""

    meta_bits = []
    if authors:
        meta_bits.append(", ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""))
    if year:
        meta_bits.append(str(year))
    if arxiv_id:
        meta_bits.append(f"arXiv:{arxiv_id}")
    meta = " · ".join(meta_bits)

    body_lines = [f"### {label}: {title}"]
    if meta:
        body_lines.append(f"_{meta}_")
    for i, c in enumerate(chunks, start=1):
        section = c.get("section") or ""
        section_tag = f" _(section: {section})_" if section else ""
        body_lines.append(f"\n**{label} chunk {i}{section_tag}:** {c['text']}")
    return "\n".join(body_lines)


def compare_papers(paper_ids: List[str]) -> Dict[str, Any]:
    if not paper_ids or len(paper_ids) < 2:
        return {
            "paper_ids": paper_ids,
            "comparison": "Provide at least two paper IDs or titles to compare.",
            "papers": [],
            "missing": [],
            "resolutions": [],
        }

    settings = get_settings()

    # ── 1. Resolve every input to a canonical paper id (handle or title search). ──
    resolutions: List[Dict[str, Any]] = []
    canonical_ids: List[Optional[str]] = []
    seen_canonical: set[str] = set()
    for raw in paper_ids:
        canonical, info = _resolve_paper_input(raw)
        # Dedupe across the input list — typing the same paper twice (or the same
        # paper as both id and title) shouldn't compare it against itself.
        if canonical:
            norm_key = canonical.lower()
            if norm_key in seen_canonical:
                info["error"] = "duplicate of another input"
                canonical = None
            else:
                seen_canonical.add(norm_key)
        info["resolved_paper_id"] = canonical
        resolutions.append(info)
        canonical_ids.append(canonical)

    if sum(1 for x in canonical_ids if x) < 2:
        return {
            "paper_ids": paper_ids,
            "papers": [],
            "missing": [r["input"] for r, c in zip(resolutions, canonical_ids) if not c],
            "resolutions": resolutions,
            "comparison": (
                "Could not resolve at least two distinct papers from the input. "
                "Try a paper id (e.g. `arxiv:2401.12345`) or a clear title — natural-language "
                "phrasing like 'compare X and Y' is fine; we'll search the index by title."
            ),
        }

    labels = [f"Paper {chr(ord('A') + i)}" for i in range(len(canonical_ids))]

    # ── 2. Build a shared anchor query so each paper's chunks align on the same theme. ──
    # We need *some* titles before we can build the anchor; do a quick first pass with a
    # generic anchor to learn each paper's title, then rebuild and refetch.
    generic_anchor = "key contributions, methods, datasets, results, limitations"
    head_chunks: Dict[str, Optional[Dict[str, Any]]] = {}
    for pid, info in zip(canonical_ids, resolutions):
        if not pid:
            continue
        # Title-search resolution already gave us a display title — skip an extra hybrid probe.
        ct = (info.get("candidate_title") or "").strip()
        if info.get("resolved_from") == "title-search" and ct:
            head_chunks[pid] = {"title": ct}
            continue
        first = _probe_one_chunk(pid, generic_anchor)
        if not first:
            first = _scroll_fallback(pid, limit=1)
        head_chunks[pid] = first[0] if first else None

    titles = [
        (head_chunks.get(pid) or {}).get("title") or ""
        for pid in canonical_ids
        if pid
    ]
    anchor = _shared_anchor_query(titles)

    # ── 3. Fetch each paper's most-anchor-relevant chunks. ──
    paper_blocks: List[str] = []
    paper_summaries: List[Dict[str, Any]] = []
    missing: List[str] = []
    for label, pid, raw in zip(labels, canonical_ids, paper_ids):
        if not pid:
            paper_blocks.append(_format_paper_block(label, []))
            paper_summaries.append(
                {
                    "label": label,
                    "paper_id_input": raw,
                    "paper_id": None,
                    "title": None,
                    "authors": [],
                    "year": None,
                    "arxiv_id": None,
                    "chunks_used": 0,
                }
            )
            missing.append(raw)
            continue

        chunks = _fetch_anchored_chunks(pid, anchor, limit=MAX_CHUNKS_PER_PAPER)
        if not chunks:
            chunks = _scroll_fallback(pid, limit=MAX_CHUNKS_PER_PAPER)
        if not chunks:
            missing.append(pid)

        paper_blocks.append(_format_paper_block(label, chunks))
        head = chunks[0] if chunks else (head_chunks.get(pid) or {})
        paper_summaries.append(
            {
                "label": label,
                "paper_id_input": raw,
                "paper_id": head.get("paper_id") or pid,
                "title": head.get("title"),
                "authors": head.get("authors") or [],
                "year": head.get("year"),
                "arxiv_id": head.get("arxiv_id"),
                "chunks_used": len(chunks),
            }
        )

    if all(s["chunks_used"] == 0 for s in paper_summaries):
        return {
            "paper_ids": paper_ids,
            "papers": paper_summaries,
            "resolutions": resolutions,
            "missing": missing,
            "comparison": (
                "None of the resolved paper IDs returned chunks from the index. "
                "The papers may not be ingested yet."
            ),
        }

    # ── 4. Synthesize the structured comparison. ──
    paper_labels_block = "\n".join(
        f"- {s['label']}: {s.get('title') or s['paper_id_input']}" for s in paper_summaries
    )
    context_block = "\n\n".join(paper_blocks)

    prompt = COMPARISON_PROMPT.format(
        paper_labels=paper_labels_block,
        context=context_block,
    )

    client = OpenAI(api_key=settings.openai_api_key)
    completion = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous research assistant. Compare papers using ONLY the provided per-paper "
                    "context. Never invent results, datasets, or numbers. Mark dimensions with no evidence as "
                    "'Not stated in retrieved context.' For every claim, attach an inline [Paper A] / [Paper B] "
                    "citation marker so the reader can audit each line."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    review = completion.choices[0].message.content or ""

    return {
        "paper_ids": paper_ids,
        "papers": paper_summaries,
        "resolutions": resolutions,
        "missing": missing,
        "comparison": review,
        "anchor_query": anchor,
    }
