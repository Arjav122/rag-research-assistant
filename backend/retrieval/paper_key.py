"""Shared, version-aware paper identity helpers.

The corpus contains a mix of legacy chunks (paper_id="arxiv:2401.12345v1") and newer
chunks ingested with the version-stripped scheme (paper_id="arxiv:2401.12345"). Without
a single normalized key, the retrieval pipeline (prefetch dedupe, diversity, citations,
recommendations, literature review) treats them as different papers — leading to the
duplicate-paper symptom seen across Search and Recommendations.

This module centralizes the normalization so every consumer agrees on what "the same
paper" means.
"""

from __future__ import annotations

import re
from typing import Any, Dict

_VERSION_SUFFIX_RE = re.compile(r"v\d+$", re.IGNORECASE)


def normalize_paper_key(meta: Dict[str, Any]) -> str:
    """Return a stable per-paper key, collapsing arxiv version suffixes.

    Resolution order:
      1. paper_id (preferred). If it carries an `arxiv:` prefix, strip the version
         suffix from the suffix portion only.
      2. arxiv_id, normalized to `arxiv:<base>` form.
      3. lowercased title (final fallback so user-uploaded PDFs without ids still dedupe).
    """
    raw_pid = (meta.get("paper_id") or "").strip()
    raw_arxiv = (meta.get("arxiv_id") or "").strip()

    if raw_pid:
        if raw_pid.startswith("arxiv:"):
            base = _VERSION_SUFFIX_RE.sub("", raw_pid[len("arxiv:") :])
            return f"arxiv:{base}"
        return _VERSION_SUFFIX_RE.sub("", raw_pid)

    if raw_arxiv:
        return f"arxiv:{_VERSION_SUFFIX_RE.sub('', raw_arxiv)}"

    return (meta.get("title") or "").strip().lower()


def chunk_paper_key(chunk: Dict[str, Any]) -> str:
    """Convenience: take a retrieval chunk and return its normalized paper key.

    Falls back to chunk id if no paper metadata is present (shouldn't happen in
    practice, but keeps callers crash-safe).
    """
    meta = chunk.get("metadata") or {}
    key = normalize_paper_key(meta)
    if key:
        return key
    return str(chunk.get("id") or "")
