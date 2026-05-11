"""Citation / sources panel rendered under chat answers.

Also exposes `linkify_citation_markers` so pages can transform inline `[n]` markers in
the LLM's markdown into anchor links that jump to the corresponding citation card.
"""

from __future__ import annotations

import html
import re
from typing import Iterable

import streamlit as st


_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _arxiv_url(arxiv_id: str) -> str:
    arxiv_id = (arxiv_id or "").strip()
    if not arxiv_id:
        return ""
    if "/" in arxiv_id or "." in arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return ""


def _authors_short(authors: list | None) -> str:
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} & {authors[1]}"
    return f"{authors[0]} et al."


def linkify_citation_markers(answer_markdown: str, available_ns: Iterable[int]) -> str:
    """Replace `[1]` / `[1,3]` / `[ 2 , 5 ]` markers with markdown links to `#cite-N`.

    Only n's that exist in `available_ns` are linkified. Unknown numbers are left as
    plain text so the user doesn't get a dangling link to nowhere.
    """
    if not answer_markdown:
        return answer_markdown
    valid = {int(n) for n in available_ns or []}

    def repl(m: re.Match) -> str:
        nums_raw = m.group(1)
        try:
            nums = [int(x.strip()) for x in nums_raw.split(",") if x.strip().isdigit()]
        except ValueError:
            return m.group(0)
        if not nums:
            return m.group(0)
        # Single-number marker: render as one link, e.g. "[1]" → "[\[1\]](#cite-1)".
        if len(nums) == 1:
            n = nums[0]
            if n in valid:
                return f"[\\[{n}\\]](#cite-{n})"
            return m.group(0)
        # Multi-number marker: link each number individually inside the brackets.
        parts = []
        for n in nums:
            if n in valid:
                parts.append(f"[{n}](#cite-{n})")
            else:
                parts.append(str(n))
        return "[" + ", ".join(parts) + "]"

    return _MARKER_RE.sub(repl, answer_markdown)


def render_citations(citations: list[dict]) -> None:
    if not citations:
        st.caption("No citations were grounded in retrieved context for this answer.")
        return

    with st.expander(f"Sources ({len(citations)})", expanded=True):
        for c in citations:
            n = c.get("n")
            title = html.escape(str(c.get("title") or "Unknown"))
            authors = _authors_short(c.get("authors") or [])
            year = c.get("year")
            src = c.get("source") or ""
            pid = c.get("paper_id") or ""
            arxiv_id = c.get("arxiv_id") or ""
            url = _arxiv_url(arxiv_id) if src == "arxiv" else ""

            meta_bits = []
            if authors:
                meta_bits.append(html.escape(authors))
            if year:
                meta_bits.append(str(year))
            if src:
                meta_bits.append(html.escape(src))
            meta = " · ".join(meta_bits)

            anchor = f"<a id='cite-{n}'></a>" if n is not None else ""
            label = f"<b>[{n}]</b>" if n is not None else "<b>•</b>"

            if url:
                tail = f"<a href='{html.escape(url)}' target='_blank'>{html.escape(arxiv_id)}</a>"
                tail = f"{tail} · {meta}" if meta else tail
            else:
                tail_bits = []
                if meta:
                    tail_bits.append(meta)
                if pid:
                    tail_bits.append(f"<code>{html.escape(pid)}</code>")
                tail = " · ".join(tail_bits)

            html_block = (
                f"<div style='padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>"
                f"{anchor}{label} {title}"
                f"<div style='font-size:0.85em;opacity:0.8;margin-top:2px;'>{tail}</div>"
                f"</div>"
            )
            st.markdown(html_block, unsafe_allow_html=True)
