from typing import List, Dict

import numpy as np
from rank_bm25 import BM25Okapi


def bm25_scores_all(query: str, documents: List[Dict]) -> np.ndarray:
    """Dense BM25 scores for every document (same order as `documents`)."""
    if not documents:
        return np.array([], dtype=float)
    tokenized_corpus = [doc["text"].split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25.get_scores(query.split())


def bm25_retrieve(query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
    if not documents:
        return []
    tokenized_corpus = [doc["text"].split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored[:top_k]]
