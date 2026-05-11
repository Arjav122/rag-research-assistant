import sys
from pathlib import Path

for _root in Path(__file__).resolve().parents:
    if (_root / "backend").is_dir() and (_root / "frontend").is_dir():
        _s = str(_root)
        if _s not in sys.path:
            sys.path.insert(0, _s)
        break

import streamlit as st

from frontend.utils.theme import inject_global_styles, page_header

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_styles()

page_header(
    "AI Research Assistant",
    "Citation-aware search, literature synthesis, and recommendations over your research corpus.",
)

st.markdown(
    """
<div class="ui-note">
Use the <b>sidebar</b> to switch workflows. Each page is focused on one research task.
</div>

| Area | Purpose |
|------|---------|
| **Research Search** | Semantic discovery over indexed papers (hybrid retrieval + cross-encoder rerank) |
| **PDF Chat** (upload page) | Drag a PDF, index, then RAG+LLM Q&A on **that file only**, with cited sources |
| **Research Chat** | Multi-turn Q&A over the full corpus with inline `[n]` citations |
| **Literature Review** | Topic synthesis from retrieved context, structured into academic sections |
| **Compare Papers** | Real chunk-grounded comparison across problem, method, datasets, results, limitations |
| **Recommendations** | Paper suggestions with per-paper rationales generated from your interest |

---
""",
    unsafe_allow_html=True,
)

st.caption(
    "Running locally? Set `BACKEND_URL` to your API (for example `http://localhost:8000`). "
    "Docker Compose wires the frontend to the `backend` service automatically."
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        '<div class="quick-card"><div class="label">Backend</div><div class="value">FastAPI + LangGraph</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="quick-card"><div class="label">Retrieval</div><div class="value">Hybrid + Cross-Encoder</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="quick-card"><div class="label">Vector Store</div><div class="value">Qdrant</div></div>',
        unsafe_allow_html=True,
    )
