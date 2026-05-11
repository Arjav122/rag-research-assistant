"""Map a query to a likely target academic section, used to bias retrieval scores.

Pure-Python keyword heuristic — no LLM call, no latency cost. If the heuristic doesn't
match (most general queries), `classify_section_intent` returns None and retrieval falls
back to its normal section-agnostic behavior.

The matched section names align with the canonical names emitted by the section-aware
chunker (see `backend.ingestion.chunker._SECTION_PATTERNS`).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Order matters: more specific intents are checked first (e.g. "ablation" should land on
# `analysis` before being shadowed by a generic "experiments" trigger).
_INTENT_RULES: List[Tuple[str, List[str]]] = [
    ("limitations", ["limitation", "weakness", "drawback", "fail", "caveat", "shortcoming", "pitfall"]),
    ("future_work", ["future work", "future research", "future direction", "next step", "open problem"]),
    ("conclusion", ["conclude", "conclusion", "takeaway", "summary of findings"]),
    ("analysis", ["ablation", "ablate", "ablation study", "analysis", "discussion"]),
    ("results", ["result", "performance", "accuracy", "benchmark score", "outperform", "state of the art",
                 "sota", "score", "improvement over", "f1", "bleu", "rouge", "metric"]),
    ("experiments", ["experiment", "experimental", "evaluation setup", "dataset", "test set",
                     "training set", "implementation detail", "hyperparameter"]),
    ("method", ["method", "methodology", "approach", "architecture", "model design",
                "algorithm", "how does", "how do they", "how is", "how are", "framework",
                "loss function", "objective", "training procedure", "pseudocode"]),
    ("introduction", ["motivation", "why", "overview of", "introduction to", "what is the problem"]),
    ("background", ["background", "preliminaries", "definition", "terminology"]),
    ("related_work", ["related work", "prior work", "literature", "previous approaches"]),
    ("abstract", ["summary of", "abstract", "tl;dr", "tldr", "in one sentence", "what is this paper"]),
]


_WORD_BOUNDARY = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    return _WORD_BOUNDARY.sub(" ", (text or "").lower()).strip()


def classify_section_intent(query: str) -> Optional[str]:
    """Return a canonical section name to boost, or None when no clear intent is found."""
    if not query:
        return None
    q = _normalize(query)
    if not q:
        return None
    padded = f" {q} "
    for section, keywords in _INTENT_RULES:
        for kw in keywords:
            kw_norm = _normalize(kw)
            if not kw_norm:
                continue
            if f" {kw_norm} " in padded:
                return section
    return None


# Sections semantically adjacent to the primary intent — also receive a smaller boost.
_NEIGHBORS = {
    "method": ["introduction", "background"],
    "results": ["experiments", "analysis"],
    "experiments": ["results", "method"],
    "analysis": ["results", "experiments"],
    "limitations": ["conclusion", "analysis"],
    "future_work": ["conclusion", "limitations"],
    "introduction": ["abstract", "background"],
    "abstract": ["introduction"],
}


def neighbor_sections(section: Optional[str]) -> List[str]:
    if not section:
        return []
    return _NEIGHBORS.get(section, [])
