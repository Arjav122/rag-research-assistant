"""Recommendation = retrieval → per-paper dedupe → MMR diversity → batched LLM rationales.

Why this exists in this shape:
- `hybrid_retrieve` returns chunk-level results with per-paper diversity already applied,
  but a recommendation list must show **papers**, not chunks. Without a second pass the
  UI was showing the same paper twice (legacy `vN` vs current id) and rec results that
  were really near-duplicates of each other.
- Cross-encoder logits (BGE-large) are unbounded and not directly comparable to vector
  similarities. The UI shows a `score`; we normalize within the result set so the value
  is interpretable as "relative confidence within these recs".
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Set, Tuple

from openai import OpenAI

from backend.prompts.templates import RECOMMENDATION_RATIONALE_PROMPT
from backend.retrieval.hybrid import hybrid_retrieve
from backend.retrieval.paper_key import chunk_paper_key
from backend.utils.config import get_settings

logger = logging.getLogger(__name__)


# Bare-minimum quality floor for the cross-encoder logit. Below this the chunk is
# essentially noise and would degrade recommendation trust if surfaced.
MIN_RECOMMENDATION_SCORE = 0.0
# MMR-lite: skip a candidate if its title+topic token-set Jaccard overlap with any
# already-selected paper exceeds this threshold. 0.65 keeps "RAG for X" and "RAG for Y"
# as distinct recommendations while collapsing near-identical re-uploads.
MMR_JACCARD_THRESHOLD = 0.65


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
_QUOTED_SPAN_RE = re.compile(r"[\"“]([^\"”]{6,200})[\"”]")
# Minimum n-gram (in tokens) the rationale must share verbatim with the snippet to
# be considered grounded. 4 is small enough that the LLM nearly always satisfies it
# when actually paraphrasing; large enough to prevent generic rubber-stamping.
MIN_GROUND_NGRAM = 4


def _tokens_lower(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _has_grounded_span(rationale: str, snippet: str) -> bool:
    """True if `rationale` shares a verbatim ≥4-token n-gram with `snippet`.

    We accept either an explicit "..." quoted span (the prompt asks for one) or any
    sufficiently long shared n-gram — robust to LLMs that drop the quote marks but
    do still copy a phrase.
    """
    if not rationale or not snippet:
        return False

    snip_tokens = _tokens_lower(snippet)
    if len(snip_tokens) < MIN_GROUND_NGRAM:
        return False
    snip_grams = {
        " ".join(snip_tokens[i : i + MIN_GROUND_NGRAM])
        for i in range(len(snip_tokens) - MIN_GROUND_NGRAM + 1)
    }

    # Prefer explicit quote, but fall back to whole-rationale n-gram match.
    quote_match = _QUOTED_SPAN_RE.search(rationale)
    candidates = []
    if quote_match:
        candidates.append(quote_match.group(1))
    candidates.append(rationale)

    for cand in candidates:
        cand_tokens = _tokens_lower(cand)
        if len(cand_tokens) < MIN_GROUND_NGRAM:
            continue
        for i in range(len(cand_tokens) - MIN_GROUND_NGRAM + 1):
            if " ".join(cand_tokens[i : i + MIN_GROUND_NGRAM]) in snip_grams:
                return True
    return False


def _title_tokens(meta: Dict) -> Set[str]:
    title = (meta.get("title") or "").lower()
    topic = (meta.get("topic") or "").lower()
    cats = " ".join(meta.get("arxiv_categories") or []).lower()
    blob = f"{title} {topic} {cats}"
    return {t for t in _TOKEN_RE.findall(blob) if len(t) > 2}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _lexical_query_overlap(query: str, item: Dict) -> float:
    """Token overlap of query (len>3) with title (2x weight) + snippet body.

    Bounded to [0,1]. Cheap proxy for "this paper is *about* the user's exact topic."
    """
    qt = {t.lower() for t in _TOKEN_RE.findall(query) if len(t) > 3}
    if not qt:
        return 0.0
    meta = item.get("metadata") or {}
    title = (meta.get("title") or "").lower()
    snip = ((item.get("text") or "")[:640]).lower()
    title_hits = sum(1 for t in qt if t in title)
    body_hits = sum(1 for t in qt if t in snip)
    # Title matches count double; cap so reranker stays primary signal.
    raw = (2.0 * title_hits + body_hits) / (3.0 * len(qt))
    return max(0.0, min(1.0, raw))


def _dedupe_keep_best(items: List[Dict]) -> List[Dict]:
    """Collapse to one chunk per paper, keeping the highest-scoring chunk."""
    by_paper: Dict[str, Dict] = {}
    for item in items:
        key = chunk_paper_key(item)
        if not key:
            continue
        score = float(item.get("rerank_score", item.get("score", 0.0)) or 0.0)
        existing = by_paper.get(key)
        if existing is None:
            by_paper[key] = item
            continue
        existing_score = float(existing.get("rerank_score", existing.get("score", 0.0)) or 0.0)
        if score > existing_score:
            by_paper[key] = item
    # Return ordered by score desc.
    return sorted(
        by_paper.values(),
        key=lambda x: float(x.get("rerank_score", x.get("score", 0.0)) or 0.0),
        reverse=True,
    )


def _apply_mmr_lite(items: List[Dict], top_k: int) -> List[Dict]:
    """Greedy MMR-lite: take the highest-scoring item, then for each next pick reject
    candidates too topically similar to anything already picked. This avoids the
    "all-five-recs-are-the-same-flavor-of-RAG" failure mode."""
    if not items:
        return []
    selected: List[Dict] = [items[0]]
    selected_tokens: List[Set[str]] = [_title_tokens(items[0].get("metadata") or {})]

    for cand in items[1:]:
        if len(selected) >= top_k:
            break
        cand_tokens = _title_tokens(cand.get("metadata") or {})
        if any(_jaccard(cand_tokens, t) >= MMR_JACCARD_THRESHOLD for t in selected_tokens):
            continue
        selected.append(cand)
        selected_tokens.append(cand_tokens)

    # If MMR was too aggressive and we ended up with fewer than top_k, backfill with
    # the next best items regardless of similarity (still honest, just less diverse).
    if len(selected) < top_k:
        seen_ids = {id(x) for x in selected}
        for cand in items[1:]:
            if len(selected) >= top_k:
                break
            if id(cand) in seen_ids:
                continue
            selected.append(cand)
    return selected[:top_k]


def _normalize_scores(items: List[Dict]) -> List[float]:
    """Min-max normalize cross-encoder logits to [0, 1] for UI display only."""
    raw = [float(x.get("rerank_score", x.get("score", 0.0)) or 0.0) for x in items]
    if not raw:
        return []
    lo, hi = min(raw), max(raw)
    if math.isclose(hi, lo):
        return [1.0 for _ in raw]
    return [(v - lo) / (hi - lo) for v in raw]


def _build_papers_block(items: List[Dict]) -> Tuple[str, List[str]]:
    """Build the papers block for the rationale prompt; return (text_block, ordered_ids)."""
    lines: List[str] = []
    ids: List[str] = []
    for item in items:
        meta = item.get("metadata") or {}
        pid = str(meta.get("paper_id") or meta.get("arxiv_id") or item.get("id") or "")
        if not pid:
            continue
        ids.append(pid)
        title = meta.get("title") or "Untitled"
        snippet = (item.get("text") or "")[:280].strip().replace("\n", " ")
        lines.append(f"{pid} | {title} | snippet: {snippet}")
    return "\n".join(lines), ids


def _parse_rationales(text: str, expected_ids: List[str]) -> Dict[str, str]:
    """Parse `paper_id :: rationale` lines emitted by the LLM. Tolerate small format drift."""
    out: Dict[str, str] = {}
    if not text:
        return out
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^[\-*\d.\s]*([^:]+?)\s*::\s*(.+)$", line)
        if not m:
            continue
        pid = m.group(1).strip().strip("`")
        why = m.group(2).strip()
        if pid:
            out[pid] = why
    if not out:
        for raw_line in text.splitlines():
            for pid in expected_ids:
                if pid and pid in raw_line:
                    rest = raw_line.split(pid, 1)[1].lstrip(" :-—|").strip()
                    if rest:
                        out[pid] = rest
                        break
    return out


def _generate_rationales(query: str, items: List[Dict]) -> Dict[str, str]:
    if not items:
        return {}
    block, ids = _build_papers_block(items)
    if not block.strip():
        return {}
    settings = get_settings()
    prompt = RECOMMENDATION_RATIONALE_PROMPT.format(query=query, papers_block=block)
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        completion = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain in one short, specific sentence per paper why it matches the user's "
                        "interest. Stay grounded in the snippet provided. No marketing tone."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = completion.choices[0].message.content or ""
    except Exception:
        logger.exception("Rationale generation failed; falling back to snippet-based reason")
        return {}

    return _parse_rationales(text, ids)


def recommend_papers(query: str, top_k: int) -> Dict:
    # Pull a wider pool than top_k so dedupe + MMR have something to work with.
    pool_size = max(top_k * 3, 12)
    pool = hybrid_retrieve(query=query, top_k=pool_size)

    # Quality floor — drop chunks the reranker thought were noise.
    pool = [
        c for c in pool
        if float(c.get("rerank_score", c.get("score", 0.0)) or 0.0) >= MIN_RECOMMENDATION_SCORE
    ]

    # One paper per chunk, then re-rank with a lexical nudge (exact-topic papers up),
    # then MMR diversification.
    deduped = _dedupe_keep_best(pool)
    settings = get_settings()
    w_lex = float(settings.recommendation_lexical_boost_weight)
    deduped.sort(
        key=lambda it: float(it.get("rerank_score", it.get("score", 0.0)) or 0.0)
        + w_lex * _lexical_query_overlap(query, it),
        reverse=True,
    )
    diversified = _apply_mmr_lite(deduped, top_k=top_k)

    if not diversified:
        return {"query": query, "recommendations": []}

    rationales = _generate_rationales(query, diversified)
    normalized_scores = _normalize_scores(diversified)

    recommendations: List[Dict] = []
    for item, norm_score in zip(diversified, normalized_scores):
        meta = item.get("metadata") or {}
        text = (item.get("text") or "")[:320].strip()
        snippet = text[:240] + ("…" if len(text) > 240 else "")
        pid = str(meta.get("paper_id") or meta.get("arxiv_id") or item.get("id") or "")
        why = rationales.get(pid)

        # Phrase-grounding gate: drop rationales that don't actually quote/borrow from
        # the snippet — that's the failure mode where the LLM produces plausible-sounding
        # but evidence-free praise. Fall back to a first-sentence echo, which is
        # guaranteed grounded since it's literally the snippet.
        full_chunk_text = item.get("text") or ""
        if why and not _has_grounded_span(why, full_chunk_text):
            why = None

        if not why:
            first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
            why = first_sentence[:220] if first_sentence else "Strong semantic match in retrieved context."
        recommendations.append(
            {
                "paper_id": meta.get("paper_id"),
                "arxiv_id": meta.get("arxiv_id"),
                "title": meta.get("title"),
                "authors": meta.get("authors") or [],
                "topic": meta.get("topic"),
                "primary_category": meta.get("primary_category"),
                "year": meta.get("year"),
                "score": round(norm_score, 3),
                "raw_rerank_score": round(
                    float(item.get("rerank_score", item.get("score", 0.0)) or 0.0), 3
                ),
                "snippet": snippet,
                "why_recommended": why,
            }
        )
    return {"query": query, "recommendations": recommendations}
