"""Recommendation cards."""

from __future__ import annotations

import streamlit as st


def render_recommendation(rec: dict, idx: int) -> None:
    title = rec.get("title") or "Untitled"
    topic = rec.get("topic") or rec.get("primary_category") or ""
    year = rec.get("year")
    score = rec.get("score")
    snippet = rec.get("snippet") or ""
    why = rec.get("why_recommended") or ""
    arxiv = rec.get("arxiv_id") or ""

    tags = " · ".join(t for t in [topic, str(year) if year else ""] if t)
    score_s = f"{float(score):.3f}" if score is not None else "—"

    with st.container():
        st.markdown(f"### {idx}. {title}")
        st.caption(f"Match score **{score_s}** · {tags}")
        if snippet:
            st.markdown(snippet)
        if why:
            st.info(why)
        if arxiv:
            st.caption(f"arXiv: `{arxiv}` · `{rec.get('paper_id','')}`")
