import re
from typing import Dict, List, Set

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def _word_set(text: str) -> Set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 2}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def trim_neighbor_redundancy(
    chunks: List[Dict],
    *,
    same_paper_jaccard: float = 0.78,
    window_chars: int = 480,
) -> List[Dict]:
    """Drop near-duplicate consecutive chunks from the **same paper** (literature review).

    Section-aware chunking often yields overlapping windows; this keeps the LLM context
    tighter without changing retrieval architecture.
    """
    if not chunks:
        return []
    out: List[Dict] = []
    prev_sig: Set[str] = set()
    prev_title = ""
    for c in chunks:
        meta = c.get("metadata") or {}
        title = str(meta.get("title") or "")
        frag = (c.get("text") or "")[:window_chars].lower()
        sig = _word_set(frag)
        if title == prev_title and prev_sig and _jaccard(sig, prev_sig) >= same_paper_jaccard:
            continue
        out.append(c)
        prev_sig = sig
        prev_title = title
    return out


def build_context(chunks: List[Dict]) -> str:
    formatted = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Unknown")
        source = meta.get("source", "unknown")
        text = chunk.get("text", "")
        formatted.append(f"[{idx}] Title: {title} | Source: {source}\n{text}")
    return "\n\n".join(formatted)
