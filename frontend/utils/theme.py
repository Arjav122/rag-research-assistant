"""Inject global SaaS-style CSS (works with Streamlit light/dark themes)."""

import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
<style>
  /* Typography & shell */
  .block-container { padding-top: 1.1rem !important; padding-bottom: 2.6rem !important; max-width: 1120px !important; }
  h1 { font-weight: 700 !important; letter-spacing: -0.02em !important; }
  h2, h3 { letter-spacing: -0.01em !important; }
  .muted { opacity: 0.72; font-size: 0.92rem; }
  .ui-note {
    border: 1px solid rgba(128,128,128,0.24);
    background: rgba(128,128,128,0.07);
    border-radius: 10px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.8rem;
    font-size: 0.9rem;
  }

  /* Cards */
  .paper-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.85rem;
    background: rgba(128,128,128,0.06);
  }
  .paper-card-title { font-size: 1.05rem; font-weight: 600; margin: 0 0 0.35rem 0; line-height: 1.35; }
  .paper-meta { font-size: 0.82rem; opacity: 0.85; margin-bottom: 0.5rem; }
  .paper-snippet { font-size: 0.9rem; line-height: 1.45; color: inherit; }
  .score-pill {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.15rem 0.45rem;
    border-radius: 999px;
    background: rgba(66, 133, 244, 0.18);
    margin-left: 0.35rem;
    vertical-align: middle;
  }
  .quick-card {
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
    background: rgba(128,128,128,0.06);
    min-height: 88px;
  }
  .quick-card .label { font-size: 0.78rem; opacity: 0.8; margin-bottom: 0.2rem; }
  .quick-card .value { font-size: 1rem; font-weight: 600; line-height: 1.35; }

  /* Chat bubbles */
  .chat-row { display: flex; margin-bottom: 0.75rem; width: 100%; }
  .chat-row.user { justify-content: flex-end; }
  .chat-row.assistant { justify-content: flex-start; }
  .bubble {
    max-width: 92%;
    padding: 0.75rem 1rem;
    border-radius: 14px;
    line-height: 1.5;
    border: 1px solid rgba(128,128,128,0.22);
  }
  .bubble.user {
    background: rgba(66, 133, 244, 0.14);
    border-bottom-right-radius: 4px;
  }
  .bubble.assistant {
    background: rgba(128,128,128,0.09);
    border-bottom-left-radius: 4px;
  }

  /* citation strip */
  .cite-strip {
    font-size: 0.8rem;
    opacity: 0.85;
    margin-top: 0.5rem;
    padding-top: 0.45rem;
    border-top: 1px dashed rgba(128,128,128,0.35);
  }

  /* Streamlit components */
  section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.2);
  }
  .stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
  }
  .stTextInput > div > div > input,
  .stTextArea textarea {
    border-radius: 10px !important;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f'<p class="muted">{subtitle}</p>', unsafe_allow_html=True)
