"""Section-aware semantic chunking with overlap; filters trivially short fragments.

Pipeline:
  1. Detect academic section boundaries (abstract / intro / methods / results / etc.)
     using simple heading regexes that work on PyMuPDF-extracted plain text.
  2. Split each section's body with RecursiveCharacterTextSplitter.
  3. Tag every emitted chunk with its enclosing section name in metadata.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.utils.config import get_settings

logger = logging.getLogger(__name__)

# Canonical section name (lowercased) → list of header regex patterns we accept.
# Patterns are intentionally permissive: optional leading numbering ("1.", "I.", "1 "),
# optional trailing punctuation, case-insensitive, must sit on its own line.
_SECTION_PATTERNS: List[Tuple[str, List[str]]] = [
    ("abstract",       [r"abstract"]),
    ("introduction",   [r"introduction", r"background and motivation", r"motivation"]),
    ("related_work",   [r"related work", r"related works", r"prior work", r"literature review"]),
    ("background",     [r"background", r"preliminaries", r"problem (formulation|statement|setup)"]),
    ("method",         [r"method(s|ology)?", r"approach", r"model", r"architecture", r"proposed method", r"framework"]),
    ("experiments",    [r"experiments?", r"experimental (setup|setting|details)", r"setup", r"implementation details"]),
    ("results",        [r"results", r"findings", r"empirical (results|evaluation)", r"evaluation"]),
    ("analysis",       [r"analysis", r"ablation(s| stud(y|ies))", r"discussion"]),
    ("limitations",    [r"limitations?", r"threats? to validity"]),
    ("conclusion",     [r"conclusions?", r"concluding remarks", r"summary"]),
    ("future_work",    [r"future work", r"future research", r"future directions"]),
    ("references",     [r"references", r"bibliography"]),
    ("acknowledgments", [r"acknowledgments?", r"acknowledgements?"]),
    ("appendix",       [r"appendix( [a-z])?", r"supplement(ary)?( material)?"]),
]

_NUMBERING = r"(?:(?:\d+(?:\.\d+)*\.?)|(?:[ivxlcdm]+\.?))"  # 1., 1.1, 2.3.4, IV., etc.


_GROUP_TO_CANON: Dict[str, str] = {}


def _build_section_regex() -> re.Pattern[str]:
    """Build one alternation regex; track group_name -> canonical_section_name."""
    alts: List[str] = []
    counter = 0
    for canon, patterns in _SECTION_PATTERNS:
        for p in patterns:
            group = f"sec{counter}"
            counter += 1
            _GROUP_TO_CANON[group] = canon
            alts.append(f"(?P<{group}>{p})")
    body = "|".join(alts)
    pattern = (
        rf"(?im)^[ \t]*"                       # line start with optional indent
        rf"(?:{_NUMBERING}[ \t]+)?"            # optional leading numbering
        rf"(?:{body})"                         # canonical heading
        rf"[ \t]*[:.\-]?[ \t]*$"               # optional trailing punctuation, end of line
    )
    return re.compile(pattern)


_SECTION_RE = _build_section_regex()


def _canon_name_from_match(m: re.Match[str]) -> str:
    for k, v in m.groupdict().items():
        if v and k in _GROUP_TO_CANON:
            return _GROUP_TO_CANON[k]
    return "body"


def detect_section_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return [(start, end, section_name), ...] covering the full text.

    A leading 'preamble' span is emitted for any text before the first detected heading,
    so we never lose chunks. If no headings are detected, returns one ('body') span.
    """
    if not text:
        return []
    matches = list(_SECTION_RE.finditer(text))
    spans: List[Tuple[int, int, str]] = []
    if not matches:
        return [(0, len(text), "body")]
    if matches[0].start() > 0:
        spans.append((0, matches[0].start(), "preamble"))
    for i, m in enumerate(matches):
        section = _canon_name_from_match(m)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if body_end > body_start:
            spans.append((body_start, body_end, section))
    return spans


def semantic_chunk_text(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Section-aware chunking. Every emitted chunk carries `section` in metadata."""
    settings = get_settings()
    if not text or len(text.strip()) < settings.ingest_min_document_chars:
        logger.debug(
            "Document text below min length (%s chars); skipping chunking",
            settings.ingest_min_document_chars,
        )
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    spans = detect_section_spans(text)
    chunks: List[Dict[str, Any]] = []
    new_idx = 0

    for start, end, section in spans:
        # Stop indexing at references/bibliography — these are noise for retrieval.
        if section in ("references", "bibliography"):
            continue
        body = text[start:end]
        if not body.strip():
            continue
        raw_chunks = splitter.split_text(body)
        for raw in raw_chunks:
            t = raw.strip()
            if len(t) < settings.ingest_min_chunk_chars:
                continue
            chunks.append(
                {
                    "text": t,
                    "metadata": {
                        **metadata,
                        "section": section,
                        "chunk_id": new_idx,
                        "chunk_char_len": len(t),
                    },
                }
            )
            new_idx += 1

    logger.debug(
        "Chunking: spans=%s kept_chunks=%s sections=%s",
        len(spans),
        len(chunks),
        sorted({c["metadata"]["section"] for c in chunks}),
    )
    return chunks


def summarize_chunk_batch(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Lightweight stats for ingestion logs / API responses."""
    if not chunks:
        return {"count": 0, "total_chars": 0, "mean_chars": 0}
    lens = [len(c["text"]) for c in chunks]
    sections: Dict[str, int] = {}
    for c in chunks:
        s = (c.get("metadata") or {}).get("section") or "body"
        sections[s] = sections.get(s, 0) + 1
    return {
        "count": len(lens),
        "total_chars": sum(lens),
        "mean_chars": round(sum(lens) / len(lens), 1),
        "min_chars": min(lens),
        "max_chars": max(lens),
        "section_counts": sections,
    }
