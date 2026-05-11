"""Loading / status helpers."""

from contextlib import contextmanager

import streamlit as st


@contextmanager
def loading_area(message: str = "Working…"):
    with st.spinner(message):
        yield


def success_banner(text: str) -> None:
    st.success(text)


def error_banner(text: str) -> None:
    st.error(text)


def info_banner(text: str) -> None:
    st.info(text)
