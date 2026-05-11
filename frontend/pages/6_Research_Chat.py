import sys
import uuid
from pathlib import Path

for _root in Path(__file__).resolve().parents:
    if (_root / "backend").is_dir() and (_root / "frontend").is_dir():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

import streamlit as st

from frontend.components.citation_panel import linkify_citation_markers, render_citations
from frontend.components.loading_components import error_banner, loading_area
from frontend.services.api_client import post
from frontend.utils.exceptions import APIError
from frontend.utils.theme import inject_global_styles, page_header

inject_global_styles()
page_header(
    "Research Chat",
    "Conversational Q&A grounded in the full indexed corpus. Answers cite the passages they used.",
)


if "research_chat_session" not in st.session_state:
    st.session_state.research_chat_session = f"research::{uuid.uuid4()}"
if "research_chat_messages" not in st.session_state:
    st.session_state.research_chat_messages = []
if "research_chat_top_k" not in st.session_state:
    st.session_state.research_chat_top_k = 8

with st.sidebar:
    st.markdown("### Chat settings")
    st.caption("Control how much context is retrieved each turn.")
    st.session_state.research_chat_top_k = int(
        min(15, max(5, st.session_state.research_chat_top_k))
    )
    st.session_state.research_chat_top_k = st.slider(
        "Passages per turn",
        min_value=5,
        max_value=15,
        value=st.session_state.research_chat_top_k,
    )
    if st.button("Reset conversation", use_container_width=True):
        try:
            post(
                f"/api/v1/chat/reset?session_id={st.session_state.research_chat_session}",
                {},
                timeout=15,
            )
        except Exception:
            pass
        st.session_state.research_chat_messages = []
        st.session_state.research_chat_session = f"research::{uuid.uuid4()}"
        st.rerun()

def _render_assistant_turn(turn: dict) -> None:
    citations = turn.get("citations") or []
    available_ns = [c.get("n") for c in citations if c.get("n") is not None]
    rendered = linkify_citation_markers(turn["content"], available_ns)
    st.markdown(rendered, unsafe_allow_html=True)
    if turn.get("low_confidence"):
        st.caption(":warning: Low retrieval confidence — try rephrasing or broadening the question.")
    if citations:
        render_citations(citations)


# Replay history
for turn in st.session_state.research_chat_messages:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant":
            _render_assistant_turn(turn)
        else:
            st.markdown(turn["content"])

prompt = st.chat_input("Ask anything about the indexed research corpus…")
if prompt:
    st.session_state.research_chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with loading_area("Retrieving context and synthesizing an answer…"):
            try:
                resp = post(
                    "/api/v1/chat/",
                    {
                        "query": prompt,
                        "session_id": st.session_state.research_chat_session,
                        "top_k": st.session_state.research_chat_top_k,
                        "retrieval_scope": "all",
                    },
                    timeout=300,
                )
                if not resp.get("success"):
                    error_banner(resp.get("message", "Chat failed"))
                else:
                    data = resp.get("data") or {}
                    answer = (data.get("answer") or "").strip() or "No relevant passages were retrieved."
                    citations = data.get("citations") or []
                    low_conf = bool(data.get("low_confidence", False))
                    turn = {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "low_confidence": low_conf,
                    }
                    _render_assistant_turn(turn)
                    st.session_state.research_chat_messages.append(turn)
            except APIError as exc:
                error_banner(str(exc))
