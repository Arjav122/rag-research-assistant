from collections import defaultdict
from typing import Dict, List

from backend.agents.workflow import build_research_graph

# Per-session memory. Cap is enforced when feeding history into the LLM (see workflow).
_memory: Dict[str, List[Dict[str, str]]] = defaultdict(list)
_graph = build_research_graph()

MAX_STORED_TURNS = 40  # rolling cap to avoid unbounded growth


def chat_with_research_assistant(
    query: str,
    session_id: str,
    top_k: int = 8,
    retrieval_scope: str = "all",
    restrict_to_paper_id: str | None = None,
) -> Dict:
    _memory[session_id].append({"role": "user", "content": query})
    if len(_memory[session_id]) > MAX_STORED_TURNS:
        _memory[session_id] = _memory[session_id][-MAX_STORED_TURNS:]

    result = _graph.invoke(
        {
            "query": query,
            "top_k": top_k,
            "retrieval_scope": retrieval_scope,
            "restrict_to_paper_id": restrict_to_paper_id or "",
            "history": list(_memory[session_id]),
        }
    )
    assistant_response = result.get("response", "")
    _memory[session_id].append({"role": "assistant", "content": assistant_response})
    if len(_memory[session_id]) > MAX_STORED_TURNS:
        _memory[session_id] = _memory[session_id][-MAX_STORED_TURNS:]

    return {
        "answer": assistant_response,
        "citations": result.get("citations", []),
        "low_confidence": bool(result.get("low_confidence", False)),
        "history": _memory[session_id],
    }


def reset_session(session_id: str) -> None:
    _memory.pop(session_id, None)
