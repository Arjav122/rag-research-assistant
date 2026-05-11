"""Controlled topic expansion: ~135 papers on focused themes (polish-phase ingestion).

For the current quality-preserving sprint (~150 papers, GraphRAG/agents/memory/eval focus),
prefer `scripts/run_controlled_corpus_expansion.py` instead.

Does NOT replace the main corpus ingest — run once after `run_ingestion.py` when you want
narrower thematic coverage without massive scaling.

Fetches **one topic per API call**. A single long OR-query often returns HTTP 200 with zero
entries (API / URL limits); per-topic calls stay reliable.

Usage:
    python scripts/run_ingestion_expansion.py

Requires .env / API keys same as main pipeline.
"""

import asyncio
import logging
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from backend.ingestion.pipeline import ingest_arxiv_pipeline
from backend.utils.config import get_settings

# One arXiv query per topic (avoid mega-OR URLs). Themes chosen for breadth + usefulness to a research assistant:
# RAG/agents/context, multimodal, trust/eval, alignment, adaptation, inference/ops.
EXPANSION_TOPICS = [
    "retrieval augmented generation large language models",
    "GraphRAG knowledge graph retrieval augmented language models",
    "dense passage retrieval neural semantic search transformers",
    "tool use autonomous agents reasoning language models planning",
    "iterative retrieval search agents LM multi-step reasoning",
    "large language model long context scaling window extrapolation",
    "language model memory episodic semantic retrieval",
    "vision language multimodal retrieval grounding augmentation",
    "hallucination detection attribution factual grounding citation language models",
    "large language model evaluation benchmarking factual correctness",
    "reinforcement learning human feedback RLHF aligned language models",
    "parameter efficient fine tuning LoRA adapters large language models",
    "quantization KV cache speculative decoding efficient inference transformers",
]

# Target ~135 new papers (~500 corpus depending on prior ingest); upserts overwrite same arxiv id (safe re-run).
MAX_RESULTS = 135


def _merge_stats(prev: dict[str, Any] | None, batch: dict[str, Any]) -> dict[str, Any]:
    if prev is None:
        return dict(batch)
    return {
        "papers_fetched": prev["papers_fetched"] + batch["papers_fetched"],
        "papers_indexed_ok": prev["papers_indexed_ok"] + batch["papers_indexed_ok"],
        "papers_skipped_already": prev.get("papers_skipped_already", 0)
        + batch.get("papers_skipped_already", 0),
        "papers_skipped": prev["papers_skipped"] + batch["papers_skipped"],
        "chunks_indexed": prev["chunks_indexed"] + batch["chunks_indexed"],
        "errors": prev["errors"] + batch["errors"],
        "chunk_stats_per_paper": prev["chunk_stats_per_paper"] + batch["chunk_stats_per_paper"],
        "qdrant_verification": batch.get("qdrant_verification") or prev.get("qdrant_verification"),
    }


async def _run_expansion() -> dict:
    log = logging.getLogger(__name__)
    settings = get_settings()
    delay = float(settings.arxiv_inter_request_delay_seconds)

    log.info("Waiting 15s before first arXiv request (rate-limit courtesy pause)...")
    await asyncio.sleep(15)

    n = len(EXPANSION_TOPICS)
    per_topic = max(1, (MAX_RESULTS + n - 1) // n)

    combined: dict[str, Any] | None = None
    for i, topic in enumerate(EXPANSION_TOPICS):
        log.info("Expansion slice %s/%s: max_results=%s topic=%r", i + 1, n, per_topic, topic)
        batch = await ingest_arxiv_pipeline(topics=[topic], max_results=per_topic)
        combined = _merge_stats(combined, batch)
        if i < n - 1:
            await asyncio.sleep(delay)

    assert combined is not None
    log.info(
        "Expansion finished: fetched=%s indexed_ok=%s skipped_already=%s chunks=%s errors=%s",
        combined["papers_fetched"],
        combined["papers_indexed_ok"],
        combined.get("papers_skipped_already", 0),
        combined["chunks_indexed"],
        len(combined["errors"]),
    )
    return combined


if __name__ == "__main__":
    result = asyncio.run(_run_expansion())
    print(result)
