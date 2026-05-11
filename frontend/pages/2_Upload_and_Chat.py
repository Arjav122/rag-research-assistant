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
from frontend.services.upload_api import upload_pdf
from frontend.utils.exceptions import APIError
from frontend.utils.theme import inject_global_styles, page_header

inject_global_styles()
page_header("Chat with your PDF", "Upload a PDF, then ask questions about it. Answers cite the passages they used.")
st.markdown(
    '<div class="ui-note">Best results come from clean research PDFs with selectable text (not scanned images).</div>',
    unsafe_allow_html=True,
)


def _session_id_for(pid: str) -> str:
    return f"pdfchat::{pid}" if pid else "pdfchat::none"


if "pdfchat_messages" not in st.session_state:
    st.session_state.pdfchat_messages = []
if "pdfchat_paper_id" not in st.session_state:
    st.session_state.pdfchat_paper_id = ""
if "pdfchat_filename" not in st.session_state:
    st.session_state.pdfchat_filename = ""

# Restore active paper on refresh.
_qp = st.query_params.get("paper_id")
if _qp and not st.session_state.pdfchat_paper_id:
    _val = _qp[0] if isinstance(_qp, list) else _qp
    if _val and _val.strip():
        st.session_state.pdfchat_paper_id = _val.strip()

st.markdown("#### Upload a PDF")
up = st.file_uploader(" ", type=["pdf"], accept_multiple_files=False, label_visibility="collapsed")

col_a, col_b = st.columns([1, 1])
with col_a:
    index_clicked = st.button(
        "Process PDF",
        type="primary",
        disabled=up is None,
        use_container_width=True,
    )
with col_b:
    reset_disabled = not st.session_state.pdfchat_paper_id and not st.session_state.pdfchat_messages
    if st.button("Reset chat", disabled=reset_disabled, use_container_width=True):
        prior_pid = st.session_state.pdfchat_paper_id
        try:
            post(
                f"/api/v1/chat/reset?session_id={_session_id_for(prior_pid)}",
                {},
                timeout=15,
            )
        except Exception:
            pass
        st.session_state.pdfchat_paper_id = ""
        st.session_state.pdfchat_filename = ""
        st.session_state.pdfchat_messages = []
        try:
            if "paper_id" in st.query_params:
                del st.query_params["paper_id"]
        except Exception:
            pass
        st.rerun()

if index_clicked and up is not None:
    raw = up.getvalue()
    with loading_area("Reading PDF and building embeddings…"):
        try:
            resp = upload_pdf(raw, up.name)
            ok_inner = (resp.get("data") or {}).get("success", True)
            if resp.get("success") and ok_inner:
                d = resp.get("data") or {}
                st.session_state.pdfchat_paper_id = (d.get("paper_id") or "").strip()
                st.session_state.pdfchat_filename = up.name
                st.session_state.pdfchat_messages = []
                if st.session_state.pdfchat_paper_id:
                    try:
                        st.query_params["paper_id"] = st.session_state.pdfchat_paper_id
                    except Exception:
                        pass
                chunks_indexed = d.get("chunks_indexed")
                stats = d.get("chunk_stats") or {}
                detail = ""
                if chunks_indexed:
                    detail = f" · indexed **{chunks_indexed}** passages"
                    if stats.get("mean_chars"):
                        detail += f" (~{int(stats['mean_chars'])} chars/passage)"
                success_banner(f"Ready. Ask anything about **{up.name}**.{detail}")
            else:
                error_banner(resp.get("message", "Upload failed"))
        except APIError as exc:
            error_banner(str(exc))

st.divider()

active_pid = (st.session_state.pdfchat_paper_id or "").strip()
if active_pid:
    fname = st.session_state.pdfchat_filename or "your PDF"
    st.caption(f"Active document: **{fname}** · `{active_pid}`")
else:
    st.info("Upload a PDF and click **Process PDF** to start chatting.")

def _render_assistant_turn(turn: dict) -> None:
    citations = turn.get("citations") or []
    available_ns = [c.get("n") for c in citations if c.get("n") is not None]
    rendered = linkify_citation_markers(turn["content"], available_ns)
    st.markdown(rendered, unsafe_allow_html=True)
    if turn.get("low_confidence"):
        st.caption(":warning: Low retrieval confidence — try rephrasing or broadening the question.")
    if citations:
        render_citations(citations)


# Replay prior turns including citations
for turn in st.session_state.pdfchat_messages:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant":
            _render_assistant_turn(turn)
        else:
            st.markdown(turn["content"])

prompt = st.chat_input(
    "Ask a question about your PDF…" if active_pid else "Upload a PDF first to start chatting",
    disabled=not active_pid,
)
if prompt and active_pid:
    st.session_state.pdfchat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with loading_area("Searching the PDF and generating an answer…"):
            try:
                resp = post(
                    "/api/v1/chat/",
                    {
                        "query": prompt,
                        "session_id": _session_id_for(active_pid),
                        "top_k": 12,
                        "retrieval_scope": "all",
                        "restrict_to_paper_id": active_pid,
                    },
                    timeout=300,
                )
                if not resp.get("success"):
                    error_banner(resp.get("message", "Chat failed"))
                else:
                    data = resp.get("data") or {}
                    answer = (data.get("answer") or "").strip() or "I couldn't find anything relevant in this PDF for that question."
                    citations = data.get("citations") or []
                    low_conf = bool(data.get("low_confidence", False))
                    turn = {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "low_confidence": low_conf,
                    }
                    _render_assistant_turn(turn)
                    st.session_state.pdfchat_messages.append(turn)
            except APIError as exc:
                error_banner(str(exc))
