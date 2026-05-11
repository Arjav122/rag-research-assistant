"""OpenAI embedding generation with batching and basic validation."""

from __future__ import annotations

import logging
from typing import List

from openai import OpenAI

from backend.utils.config import get_settings

logger = logging.getLogger(__name__)

# text-embedding-3-large default dimension
EXPECTED_DIM_LARGE = 3072


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    model = settings.openai_embedding_model
    batch_size = max(1, settings.embedding_batch_size)

    all_vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        ordered = sorted(response.data, key=lambda x: x.index)
        vectors = [item.embedding for item in ordered]
        all_vectors.extend(vectors)

        if start == 0 and vectors:
            dim = len(vectors[0])
            logger.info("Embedding model=%s first_batch_dim=%s batch_size=%s", model, dim, len(batch))
            if model.endswith("large") and dim != EXPECTED_DIM_LARGE:
                logger.warning(
                    "Unexpected embedding dimension %s for %s (expected %s for 3-large default)",
                    dim,
                    model,
                    EXPECTED_DIM_LARGE,
                )

    if len(all_vectors) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: got {len(all_vectors)} for {len(texts)} texts")

    return all_vectors
