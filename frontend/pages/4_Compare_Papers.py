import sys
from pathlib import Path

for _root in Path(__file__).resolve().parents:
    if (_root / "backend").is_dir() and (_root / "frontend").is_dir():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

import streamlit as st

from frontend.components.loading_components import error_banner, loading_area, info_banner, success_banner
from frontend.services.api_client import post
from frontend.utils.exceptions import APIError
from frontend.utils.theme import inject_global_styles, page_header

inject_global_styles()
page_header(
    "Compare papers",
    "Paste paper IDs OR titles — natural-language phrasing is fine. We'll resolve titles "
    "by searching the index, then synthesise a chunk-grounded comparison.",
)
st.markdown(
    '<div class="ui-note">Use 2-5 papers for clearer comparisons. You can mix paper titles and arXiv-style IDs.</div>',
    unsafe_allow_html=True,
)

paper_ids_raw = st.text_area(
    "Papers to compare (one per line, or comma-separated)",
    placeholder=(
        "LatentRAG\n"
        "Superintelligent Retrieval Agent\n\n"
        "or paste IDs directly:\n"
        "arxiv:2401.12345, arxiv:2401.12346"
    ),
    height=140,
)

go = st.button("Compare", type="primary", use_container_width=True)

if go and paper_ids_raw.strip():
    parts: list[str] = []
    for line in paper_ids_raw.replace(",", "\n").splitlines():
        x = line.strip()
        if x:
            parts.append(x)

    if len(parts) < 2:
        error_banner("Enter at least two papers — IDs or titles, one per line.")
    else:
        with loading_area("Resolving papers, fetching aligned chunks, synthesising comparison…"):
            try:
                resp = post("/api/v1/comparison/", {"paper_ids": parts}, timeout=300)
                if not resp.get("success"):
                    error_banner(resp.get("message", "Failed"))
                else:
                    data = resp.get("data") or {}
                    comp = data.get("comparison") or ""
                    papers = data.get("papers") or []
                    missing = data.get("missing") or []
                    resolutions = data.get("resolutions") or []
                    anchor_query = data.get("anchor_query") or ""

                    indexed_count = sum(1 for p in papers if p.get("chunks_used", 0) > 0)
                    success_banner(
                        f"Resolved **{indexed_count}/{len(papers) or len(parts)}** papers from the index."
                    )

                    if resolutions:
                        st.markdown("#### Input resolution")
                        for r in resolutions:
                            inp = r.get("input") or ""
                            via = r.get("resolved_from")
                            resolved = r.get("resolved_paper_id")
                            cand_title = r.get("candidate_title")
                            score = r.get("candidate_score")
                            err = r.get("error")
                            if resolved and via == "title-search":
                                st.markdown(
                                    f"- `{inp}` → **{cand_title or resolved}** "
                                    f"(`{resolved}`, match score {score})"
                                )
                            elif resolved and via == "handle":
                                st.markdown(f"- `{inp}` → `{resolved}` (direct id)")
                            else:
                                st.markdown(f"- `{inp}` → _unresolved_ ({err or 'no match'})")
                        st.divider()

                    if papers:
                        st.markdown("#### Papers under comparison")
                        for p in papers:
                            label = p.get("label") or "?"
                            title = p.get("title") or p.get("paper_id_input") or "(unknown)"
                            year = p.get("year")
                            authors = p.get("authors") or []
                            authors_s = ""
                            if authors:
                                authors_s = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
                            chunks = p.get("chunks_used", 0)
                            arxiv_id = p.get("arxiv_id") or ""
                            meta_bits = [b for b in [authors_s, str(year) if year else "", arxiv_id] if b]
                            line = f"**{label}** — {title}"
                            if meta_bits:
                                line += "  \n" + " · ".join(meta_bits)
                            line += f"  \nChunks used: **{chunks}**"
                            if chunks == 0:
                                line += " — _not found in index_"
                            st.markdown(line)
                        st.divider()

                    if missing:
                        info_banner(
                            "Some inputs were not found in the index: "
                            + ", ".join(f"`{m}`" for m in missing)
                            + ". They appear as 'Not stated in retrieved context.' in the comparison."
                        )

                    if anchor_query:
                        st.caption(f"Shared retrieval anchor: _{anchor_query}_")

                    if comp:
                        st.markdown(comp)
            except APIError as exc:
                error_banner(str(exc))
