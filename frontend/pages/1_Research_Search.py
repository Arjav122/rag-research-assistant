import sys
from pathlib import Path

for _root in Path(__file__).resolve().parents:
    if (_root / "backend").is_dir() and (_root / "frontend").is_dir():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

import streamlit as st

from frontend.components.loading_components import error_banner, loading_area, success_banner
from frontend.components.paper_card import render_paper_card
from frontend.services.api_client import post
from frontend.utils.exceptions import APIError
from frontend.utils.theme import inject_global_styles, page_header

inject_global_styles()
page_header("Research Search", "Semantic + keyword fusion with cross-encoder reranking.")
st.markdown(
    '<div class="ui-note">Tip: include method names, benchmarks, or acronyms to improve precision (example: "LoRA on MMLU with retrieval").</div>',
    unsafe_allow_html=True,
)

query = st.text_input("Search query", placeholder="e.g. retrieval augmented generation for LLMs")
top_k = st.slider("Number of results", 5, 15, 10)

if st.button("Search", type="primary", use_container_width=True) and query.strip():
    with loading_area("Searching corpus…"):
        try:
            resp = post("/api/v1/search/", {"query": query.strip(), "top_k": top_k})
            if not resp.get("success"):
                error_banner(resp.get("message", "Request failed"))
            else:
                data = resp.get("data") or {}
                results = data.get("results") or []
                success_banner(f"Found **{len(results)}** ranked passages.")
                if not results:
                    st.warning("No hits — try broader terms or confirm ingestion has run.")
                for i, hit in enumerate(results, start=1):
                    render_paper_card(hit, rank=i, expanded_default=(i <= 3))
        except APIError as exc:
            error_banner(str(exc))
