from backend.retrieval.hybrid import hybrid_retrieve


def semantic_search(query: str, top_k: int) -> dict:
    results = hybrid_retrieve(query=query, top_k=top_k)
    return {"query": query, "results": results}
