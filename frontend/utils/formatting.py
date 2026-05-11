"""Format API payloads for display."""


def authors_str(authors: object, limit: int = 5) -> str:
    if not authors:
        return ""
    if isinstance(authors, str):
        return authors
    if isinstance(authors, list):
        parts = [str(a).strip() for a in authors[:limit] if a]
        if len(authors) > limit:
            parts.append("…")
        return ", ".join(parts)
    return str(authors)


def snippet_from_text(text: str, max_len: int = 280) -> str:
    t = " ".join((text or "").split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rsplit(" ", 1)[0] + "…"


def score_display(hit: dict) -> str:
    if hit.get("rerank_score") is not None:
        return f"{float(hit['rerank_score']):.3f}"
    if hit.get("score") is not None:
        return f"{float(hit['score']):.4f}"
    return "—"
