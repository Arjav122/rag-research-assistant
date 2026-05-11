"""
Automated API smoke test for CI / pre-demo checks.

Usage (from project root, with backend reachable):
  python scripts/smoke_test_api.py
  python scripts/smoke_test_api.py --base http://localhost:8000 --quick

Environment:
  SMOKE_BASE_URL              default http://localhost:8000
  SMOKE_TIMEOUT_SEARCH        default 300
  SMOKE_TIMEOUT_CHAT          default 300
  SMOKE_TIMEOUT_COMPARISON     default 300
  SMOKE_TIMEOUT_RECOMMENDATION default 300
  SMOKE_TIMEOUT_LITERATURE     default 600
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default=os.environ.get("SMOKE_BASE_URL", "http://localhost:8000"),
        help="API base URL",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Skip literature review only (all other endpoints use full timeouts)",
    )
    ap.add_argument(
        "--timeout-search",
        type=int,
        default=int(os.environ.get("SMOKE_TIMEOUT_SEARCH", "300")),
        help="Seconds for search endpoint (reranker cold start can exceed 120s)",
    )
    ap.add_argument(
        "--timeout-comparison",
        type=int,
        default=int(os.environ.get("SMOKE_TIMEOUT_COMPARISON", "300")),
        help="Seconds for comparison (two hybrid_retrieve calls per paper)",
    )
    ap.add_argument(
        "--timeout-chat",
        type=int,
        default=int(os.environ.get("SMOKE_TIMEOUT_CHAT", "300")),
        help="Seconds for chat (LangGraph + retrieval + LLM; cold rerank can exceed 90s)",
    )
    ap.add_argument(
        "--timeout-recommendation",
        type=int,
        default=int(os.environ.get("SMOKE_TIMEOUT_RECOMMENDATION", "300")),
        help="Seconds for recommendation endpoint",
    )
    ap.add_argument(
        "--timeout-literature",
        type=int,
        default=int(os.environ.get("SMOKE_TIMEOUT_LITERATURE", "600")),
        help="Seconds for literature review (long-running synthesis)",
    )
    args = ap.parse_args()
    base = args.base.rstrip("/")

    try:
        import requests
    except ImportError:
        print("ERROR: install requests: pip install requests", file=sys.stderr)
        return 2

    fails: list[str] = []
    session = requests.Session()

    def ok(name: str, fn) -> None:
        t0 = time.perf_counter()
        try:
            fn()
            ms = (time.perf_counter() - t0) * 1000
            log(f"  OK  {name}  ({ms:.0f} ms)")
        except Exception as e:
            fails.append(f"{name}: {e}")
            ms = (time.perf_counter() - t0) * 1000
            log(f"  FAIL {name}  ({ms:.0f} ms)  -> {e}")

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"Smoke test against {base}\n")

    def check_health():
        r = session.get(f"{base}/health", timeout=60)
        r.raise_for_status()
        assert r.json().get("status") == "ok"

    def check_root():
        r = session.get(f"{base}/", timeout=30)
        r.raise_for_status()
        js = r.json()
        assert js.get("service") == "ai-research-assistant-backend"
        assert "/api/v1" in (js.get("api_prefix") or "")

    def check_search():
        r = session.post(
            f"{base}/api/v1/search/",
            json={"query": "retrieval augmented generation", "top_k": 5},
            timeout=args.timeout_search,
        )
        r.raise_for_status()
        body = r.json()
        assert body.get("success") is True
        data = body.get("data") or {}
        results = data.get("results") or []
        assert len(results) >= 1, "no search results"

    def check_recommendation():
        r = session.post(
            f"{base}/api/v1/recommendation/",
            json={"query": "RAG agents", "top_k": 3},
            timeout=args.timeout_recommendation,
        )
        r.raise_for_status()
        body = r.json()
        assert body.get("success") is True
        recs = (body.get("data") or {}).get("recommendations") or []
        assert len(recs) >= 1, "no recommendations"

    def check_chat():
        sid = f"smoke::{uuid.uuid4()}"
        r = session.post(
            f"{base}/api/v1/chat/",
            json={
                "query": "What is retrieval-augmented generation in one sentence?",
                "session_id": sid,
                "top_k": 6,
                "retrieval_scope": "all",
            },
            timeout=args.timeout_chat,
        )
        r.raise_for_status()
        body = r.json()
        assert body.get("success") is True
        ans = (body.get("data") or {}).get("answer") or ""
        assert len(ans) > 20, "empty or tiny answer"

    def check_chat_reset():
        sid = f"smoke-reset::{uuid.uuid4()}"
        r = session.post(
            f"{base}/api/v1/chat/reset",
            params={"session_id": sid},
            timeout=30,
        )
        r.raise_for_status()
        body = r.json()
        assert body.get("success") is True
        sid_back = (body.get("data") or {}).get("session_id")
        assert sid_back == sid

    def check_comparison():
        r = session.post(
            f"{base}/api/v1/comparison/",
            json={"paper_ids": ["LatentRAG", "Superintelligent Retrieval Agent"]},
            timeout=args.timeout_comparison,
        )
        r.raise_for_status()
        body = r.json()
        assert body.get("success") is True
        comp = (body.get("data") or {}).get("comparison") or ""
        assert len(comp) > 50, "empty comparison"

    def check_literature():
        r = session.post(
            f"{base}/api/v1/literature/review",
            json={"topic": "Retrieval-augmented generation for language models", "max_papers": 8},
            timeout=args.timeout_literature,
        )
        r.raise_for_status()
        body = r.json()
        assert body.get("success") is True
        rev = (body.get("data") or {}).get("review") or ""
        assert len(rev) > 100, "empty literature review"

    log("Checks:")
    ok("GET /health", check_health)
    ok("GET /", check_root)
    ok("POST /api/v1/search/", check_search)
    ok("POST /api/v1/recommendation/", check_recommendation)
    ok("POST /api/v1/chat/", check_chat)
    ok("POST /api/v1/chat/reset", check_chat_reset)
    ok("POST /api/v1/comparison/", check_comparison)
    if not args.quick:
        ok("POST /api/v1/literature/review", check_literature)
    else:
        log("  SKIP literature/review (--quick)")

    if fails:
        log("\n--- FAILURES ---")
        for f in fails:
            log(f"  {f}")
        return 1
    log("\nAll checks passed. Ready for your manual UI pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
