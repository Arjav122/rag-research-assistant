import sys
from pathlib import Path

for _root in Path(__file__).resolve().parents:
    if (_root / "backend").is_dir() and (_root / "frontend").is_dir():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

import streamlit as st

from frontend.components.loading_components import error_banner, loading_area, success_banner
from frontend.components.recommendation_card import render_recommendation
from frontend.services.api_client import post
from frontend.utils.exceptions import APIError
from frontend.utils.theme import inject_global_styles, page_header

inject_global_styles()
page_header("Recommendations", "Papers whose passages best match your research interest profile.")
st.markdown(
    '<div class="ui-note">Add constraints like task, domain, or model family for more relevant recommendations.</div>',
    unsafe_allow_html=True,
)

query = st.text_input("Interest / topic", placeholder="e.g. RLHF policy optimization for LLMs")
top_k = st.slider("How many suggestions", 5, 15, 8)

if st.button("Recommend", type="primary", use_container_width=True) and query.strip():
    with loading_area("Ranking papers…"):
        try:
            resp = post("/api/v1/recommendation/", {"query": query.strip(), "top_k": top_k})
            if not resp.get("success"):
                error_banner(resp.get("message", "Failed"))
            else:
                data = resp.get("data") or {}
                recs = data.get("recommendations") or []
                success_banner(f"**{len(recs)}** recommendations.")
                if not recs:
                    st.warning("No recommendations — broaden your query or check ingestion.")
                for i, rec in enumerate(recs, start=1):
                    render_recommendation(rec, i)
                    st.divider()
        except APIError as exc:
            error_banner(str(exc))
