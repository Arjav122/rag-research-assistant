from typing import TypedDict, List, Dict


class AgentState(TypedDict, total=False):
    query: str
    top_k: int
    retrieval_scope: str
    restrict_to_paper_id: str
    history: List[Dict[str, str]]
    context_chunks: List[Dict]
    context: str
    response: str
    citations: List[Dict]
    low_confidence: bool
