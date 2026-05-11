import sys
from pathlib import Path

for _root in Path(__file__).resolve().parents:
    if (_root / "backend").is_dir() and (_root / "frontend").is_dir():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

import streamlit as st

from frontend.components.citation_panel import linkify_citation_markers, render_citations
from frontend.components.loading_components import error_banner, loading_area, success_banner
from frontend.services.api_client import post
from frontend.utils.exceptions import APIError
from frontend.utils.theme import inject_global_styles, page_header

inject_global_styles()
page_header(
    "Literature review",
    "Structured academic synthesis grounded in retrieved passages, with inline citations.",
)
st.markdown(
    '<div class="ui-note">Choose a focused topic phrase for cleaner, technically specific reviews.</div>',
    unsafe_allow_html=True,
)

topic = st.text_input("Review topic", placeholder="e.g. scaling laws for language models")
max_papers = st.slider("Context passages to retrieve", 5, 15, 10)

if st.button("Generate review", type="primary", use_container_width=True) and topic.strip():
    with loading_area("Gathering context and drafting structured review…"):
        try:
            resp = post(
                "/api/v1/literature/review",
                {"topic": topic.strip(), "max_papers": max_papers},
                timeout=300,
            )
            if not resp.get("success"):
                error_banner(resp.get("message", "Failed"))
            else:
                data = resp.get("data") or {}
                review = data.get("review") or ""
                nsrc = data.get("sources", 0)
                npapers = data.get("papers", 0)
                citations = data.get("citations") or []
                low_conf = bool(data.get("low_confidence", False))

                success_banner(
                    f"Synthesized from **{nsrc}** passages spanning **{npapers}** papers."
                )
                if low_conf:
                    st.warning(
                        "Retrieved evidence is weak for this topic. Consider broader phrasing or different terminology."
                    )

                available_ns = [c.get("n") for c in citations if c.get("n") is not None]
                st.markdown(
                    linkify_citation_markers(review, available_ns),
                    unsafe_allow_html=True,
                )

                if citations:
                    render_citations(citations)

                st.download_button(
                    "Download as Markdown",
                    review,
                    file_name="literature_review.md",
                    mime="text/markdown",
                )
        except APIError as exc:
            error_banner(str(exc))
