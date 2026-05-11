"""Optional Qdrant payload filters for scoped retrieval (e.g. user uploads only)."""

from __future__ import annotations

from typing import Any, List, Optional

from qdrant_client.http.models import Filter, FieldCondition, MatchValue


def build_retrieval_filter(
    retrieval_scope: str = "all",
    restrict_to_paper_id: Optional[str] = None,
) -> Optional[Filter]:
    """
    retrieval_scope:
      - \"all\" — no filter (full corpus)
      - \"user_uploads\" — only chunks from uploaded PDFs (payload.source == user_upload)
    restrict_to_paper_id:
      - If set (e.g. \"user:<uuid>\"), only that paper's chunks (implies scoped retrieval).
    """
    conditions: List[Any] = []

    if restrict_to_paper_id and restrict_to_paper_id.strip():
        conditions.append(
            FieldCondition(key="paper_id", match=MatchValue(value=restrict_to_paper_id.strip()))
        )
    elif retrieval_scope == "user_uploads":
        conditions.append(
            FieldCondition(key="source", match=MatchValue(value="user_upload"))
        )

    if not conditions:
        return None

    return Filter(must=conditions)
