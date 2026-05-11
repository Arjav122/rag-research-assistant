"""CLI: arXiv ingestion. Defaults match the broadened topic list in `IngestRequest`."""

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from backend.ingestion.pipeline import ingest_arxiv_pipeline

DEFAULT_TOPICS = [
    "Large Language Models",
    "Retrieval Augmented Generation",
    "Natural Language Processing",
    "RLHF",
    "Instruction Tuning",
    "Alignment",
    "Chain of Thought Reasoning",
    "AI Agents",
    "Tool Use",
    "Multimodal Learning",
    "Vision Language Models",
    "Diffusion Models",
    "LLM Evaluation",
    "Benchmarks",
    "Mixture of Experts",
    "Long Context",
]


if __name__ == "__main__":
    result = asyncio.run(
        ingest_arxiv_pipeline(
            topics=DEFAULT_TOPICS,
            max_results=300,
        )
    )
    print(result)
