"""Tier 1 retrieval evaluation: structured probe set hitting the live /chat endpoint.

Each probe targets a specific failure mode we previously identified. For each call we
capture: latency, citations, low_confidence flag, inline [n] marker presence, and the
top-cited paper title — so the report tells us *where retrieval actually picks the
right paper*, not just whether the LLM produced text.

Run inside the backend container so we hit FastAPI on its private network:
  docker exec ai_research_backend python scripts/evaluation_probe.py
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

BACKEND = "http://localhost:8000"

PROBES: List[Dict[str, Any]] = [
    {
        "id": "P01_acronym",
        "category": "Acronym recall",
        "query": "RAG",
        "expect": "Should expand RAG → Retrieval-Augmented Generation; HyDE should activate; multi-paper answer with citations.",
    },
    {
        "id": "P02_lexical",
        "category": "Lexical / exact-token",
        "query": "FAISS",
        "expect": "BM25 corpus path should surface papers literally mentioning FAISS; static expansion adds 'Facebook AI Similarity Search'.",
    },
    {
        "id": "P03_method",
        "category": "Methodology question",
        "query": "What is the architecture of LatentRAG?",
        "expect": "section_intent → 'method'; should retrieve method-section chunks; cite the LatentRAG paper.",
    },
    {
        "id": "P04_results",
        "category": "Results question",
        "query": "What accuracy improvements does GATHER report?",
        "expect": "section_intent → 'results'; should cite the GATHER cell-type annotation paper.",
    },
    {
        "id": "P05_limitations",
        "category": "Limitations question",
        "query": "What are the limitations of agentic RAG approaches?",
        "expect": "section_intent → 'limitations'; should retrieve limitation-section chunks across multiple papers.",
    },
    {
        "id": "P06_comparison",
        "category": "Multi-entity / comparison",
        "query": "Compare RLHF and DPO",
        "expect": "Both abbreviations expanded; retrieval may favor whichever has more chunks; tests the multi-entity weakness.",
    },
    {
        "id": "P07_offdomain",
        "category": "Off-domain (should refuse / low-confidence)",
        "query": "How does CRISPR gene editing work?",
        "expect": "low_confidence=True; reranker scores should be poor; answer should NOT confidently invent biology.",
    },
    {
        "id": "P08_title_only",
        "category": "Title-only retrieval",
        "query": "the LatentRAG paper",
        "expect": "Without title-as-vector this may still work via body mentions; tests how badly the gap hurts.",
    },
    {
        "id": "P09_recency",
        "category": "Recency / year-scoped",
        "query": "Latest 2026 papers on multimodal language models",
        "expect": "All papers are 2026, so this should work — but tests that 'latest' / 'multimodal' doesn't hallucinate.",
    },
    {
        "id": "P10_multiturn_a",
        "category": "Multi-turn (turn 1)",
        "query": "What is Mixture of Experts?",
        "expect": "Baseline turn — sets context for follow-up.",
        "session_tag": "multiturn",
    },
    {
        "id": "P10_multiturn_b",
        "category": "Multi-turn follow-up (turn 2)",
        "query": "What about its limitations?",
        "expect": "Query rewriter must self-contain this using prior history → 'limitations of Mixture of Experts'.",
        "session_tag": "multiturn",
    },
]


def call_chat(query: str, session_id: str, top_k: int = 6) -> Dict[str, Any]:
    t0 = time.monotonic()
    try:
        r = requests.post(
            f"{BACKEND}/api/v1/chat/",
            json={
                "query": query,
                "session_id": session_id,
                "top_k": top_k,
                "retrieval_scope": "all",
            },
            timeout=240,
        )
        r.raise_for_status()
        data = r.json().get("data") or {}
    except Exception as exc:
        return {"error": str(exc), "elapsed_s": round(time.monotonic() - t0, 1)}
    return {
        "answer": data.get("answer") or "",
        "citations": data.get("citations") or [],
        "low_confidence": bool(data.get("low_confidence", False)),
        "elapsed_s": round(time.monotonic() - t0, 1),
    }


_INLINE_CITE_RE = re.compile(r"\[\s*(\d+)(?:\s*,\s*\d+)*\s*\]")


def inline_marker_count(answer: str) -> int:
    return len(_INLINE_CITE_RE.findall(answer or ""))


def truncate(text: str, n: int = 280) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> None:
    print(f"\n{'=' * 80}\n  TIER 1 RETRIEVAL EVALUATION\n{'=' * 80}")
    sessions: Dict[str, str] = {}
    results: List[Dict[str, Any]] = []

    for probe in PROBES:
        tag = probe.get("session_tag")
        if tag:
            sid = sessions.setdefault(tag, f"eval-{tag}-{uuid.uuid4().hex[:6]}")
        else:
            sid = f"eval-{probe['id']}-{uuid.uuid4().hex[:6]}"

        print(f"\n--- {probe['id']} | {probe['category']} ---")
        print(f"  query   : {probe['query']!r}")
        print(f"  expect  : {probe['expect']}")

        out = call_chat(probe["query"], session_id=sid)
        if "error" in out:
            print(f"  ERROR   : {out['error']}")
            results.append({**probe, **out, "session_id": sid})
            continue

        answer = out["answer"]
        cites = out["citations"]
        n_inline = inline_marker_count(answer)
        cite_titles = [c.get("title") for c in cites]

        print(f"  latency : {out['elapsed_s']}s")
        print(f"  inline_[n]: {n_inline}    citations_returned: {len(cites)}    low_confidence: {out['low_confidence']}")
        if cite_titles:
            for i, t in enumerate(cite_titles[:5], start=1):
                print(f"     [{i}] {t}")
        print(f"  answer  : {truncate(answer, 320)}")

        results.append({**probe, **out, "session_id": sid, "inline_markers": n_inline})

    # ----- summary -----
    print(f"\n{'=' * 80}\n  SUMMARY\n{'=' * 80}")
    rows = []
    for r in results:
        if "error" in r:
            rows.append((r["id"], r["category"], "ERROR", "-", "-", "-", "-"))
            continue
        rows.append(
            (
                r["id"],
                r["category"],
                f"{r['elapsed_s']}s",
                str(len(r["citations"])),
                str(r.get("inline_markers", 0)),
                "YES" if r["low_confidence"] else "no",
                r["citations"][0].get("title", "")[:60] if r["citations"] else "—",
            )
        )

    headers = ("id", "category", "lat", "cites", "[n]", "lowconf", "top citation")
    widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


if __name__ == "__main__":
    main()
