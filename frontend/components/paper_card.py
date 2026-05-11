"""Paper / search hit cards."""

from __future__ import annotations

import streamlit as st

from frontend.utils.formatting import authors_str, score_display, snippet_from_text


def render_paper_card(hit: dict, rank: int | None = None, expanded_default: bool = False) -> None:
    meta = hit.get("metadata") or {}
    title = meta.get("title") or "Untitled"
    authors = authors_str(meta.get("authors")) or "—"
    year = meta.get("year") if meta.get("year") is not None else "—"
    topic = meta.get("topic") or meta.get("primary_category") or "—"
    arxiv_id = meta.get("arxiv_id") or ""
    paper_id = meta.get("paper_id") or ""
    text = hit.get("text") or ""
    snippet = snippet_from_text(text, 320)
    score = score_display(hit)

    label = f"{f'#{rank} · ' if rank is not None else ''}{title}"
    with st.expander(f"{label} · score {score}", expanded=expanded_default):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption(f"**Authors:** {authors}")
            st.caption(f"**Year:** {year} · **Topic:** {topic}")
        with c2:
            st.metric("Relevance", score)
        st.markdown(snippet)
        st.caption(f"`paper_id`: {paper_id}" + (f" · `arxiv`: `{arxiv_id}`" if arxiv_id else ""))
