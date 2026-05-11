"""CLI: hybrid retrieval smoke test (requires vectors in Qdrant + OPENAI_API_KEY)."""

import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

from backend.retrieval.hybrid import hybrid_retrieve


def main() -> None:
    query = "retrieval augmented generation large language models"
    hits = hybrid_retrieve(query=query, top_k=5)
    simplified = []
    for h in hits:
        meta = h.get("metadata") or {}
        simplified.append(
            {
                "score": h.get("score"),
                "title": meta.get("title"),
                "arxiv_id": meta.get("arxiv_id"),
                "text_preview": (h.get("text") or "")[:300],
            }
        )
    print(json.dumps({"query": query, "hits": simplified}, indent=2))


if __name__ == "__main__":
    main()
