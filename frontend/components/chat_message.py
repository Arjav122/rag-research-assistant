"""Chat-style rendering using Streamlit chat widgets."""

from __future__ import annotations

import streamlit as st


def render_turn(role: str, content: str) -> None:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)


def render_history(history: list[dict]) -> None:
    for turn in history:
        render_turn(turn.get("role", "user"), turn.get("content") or "")
