from typing import List, Dict
import logging
import time
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_model = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder("BAAI/bge-reranker-large")
    return _model


def rerank(query: str, candidates: List[Dict], top_k: int = 8) -> List[Dict]:
    if not candidates:
        return []
    t0 = time.monotonic()
    model = get_reranker()
    pairs = [[query, item["text"]] for item in candidates]
    # CPU-only Docker: large batches + tqdm progress add latency; keep batch modest.
    scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
    scored = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
    out: List[Dict] = []
    for cand, score in scored[:top_k]:
        row = dict(cand)
        row["rerank_score"] = float(score)
        out.append(row)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "Rerank timing: candidates=%s top_k=%s elapsed_ms=%s",
        len(candidates),
        top_k,
        elapsed_ms,
    )
    return out
