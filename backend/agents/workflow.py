import re
import logging
import time
from typing import Dict, List

from langgraph.graph import StateGraph, END
from openai import OpenAI

from backend.agents.state import AgentState
from backend.prompts.templates import RETRIEVAL_QA_PROMPT
from backend.retrieval.context_builder import build_context
from backend.retrieval.hybrid import hybrid_retrieve
from backend.retrieval.paper_key import normalize_paper_key
from backend.retrieval.qdrant_filters import build_retrieval_filter
from backend.utils.config import get_settings

logger = logging.getLogger(__name__)


# Limit how much chat history we feed back into the LLM. Older turns are summarized away
# implicitly by truncation; the prompt still accepts plain role-tagged transcript.
HISTORY_TURN_BUDGET = 6
HISTORY_CHAR_BUDGET = 4000


def _max_rerank_score(chunks: List[Dict]) -> float:
    if not chunks:
        return float("-inf")
    scores = [float(c.get("rerank_score", c.get("score", 0.0)) or 0.0) for c in chunks]
    return max(scores) if scores else float("-inf")


# Catches LLM disclaimers when retrieval matched on adjacent vocabulary but the corpus
# doesn't actually answer the question (e.g. "How does CRISPR work?" against a corpus
# that only has retinal-imaging papers mentioning gene editing). The reranker gives
# misleadingly high scores in that case, so the threshold-based guardrail can't fire,
# but the LLM correctly says it can't answer — and we should drop the citation chips
# accordingly so the UI matches the answer's honesty.
_DISCLAIMER_PATTERNS = re.compile(
    r"(context (?:blocks? )?(?:do(?:es)? not |don't |doesn't )(?:contain|include|provide|cover|address|discuss|mention)|"
    r"(?:no|insufficient|not enough) (?:relevant )?(?:information|context|evidence|details|content|coverage)|"
    r"(?:cannot|can't|unable to) (?:answer|address|determine|conclude|infer)|"
    r"the (?:provided |retrieved )?context (?:is|are) (?:insufficient|not sufficient|inadequate)|"
    r"(?:i|we) (?:cannot|can't|do not have|don't have) (?:enough )?(?:information|context|evidence)|"
    r"limited evidence in the indexed corpus)",
    re.IGNORECASE,
)


def _answer_is_disclaimer(answer: str) -> bool:
    """True if the LLM's own answer signals the corpus didn't actually address the
    question. Used to suppress citation chips that would otherwise look misleading
    next to a disclaiming answer."""
    if not answer:
        return False
    head = answer[:600]  # disclaimers are virtually always near the top
    return bool(_DISCLAIMER_PATTERNS.search(head))


def retrieval_agent(state: AgentState) -> AgentState:
    """Run hybrid retrieval and pre-compute the confidence signal so downstream nodes
    can suppress citations / soften the answer prompt under weak evidence."""
    settings = get_settings()
    scope = state.get("retrieval_scope") or "all"
    paper_id = (state.get("restrict_to_paper_id") or "").strip() or None
    qf = build_retrieval_filter(retrieval_scope=scope, restrict_to_paper_id=paper_id)
    k = int(state.get("top_k") or 8)
    chunks = hybrid_retrieve(
        query=state["query"],
        top_k=k,
        qdrant_filter=qf,
        history=state.get("history") or None,
    )
    state["context_chunks"] = chunks

    low_confidence = False
    if settings.retrieval_use_confidence_guardrail and chunks:
        low_confidence = (
            _max_rerank_score(chunks) < settings.retrieval_low_confidence_threshold
        )
    state["low_confidence"] = bool(low_confidence)
    return state


def citation_agent(state: AgentState) -> AgentState:
    """Emit one citation per paper, numbered to match Context block ordering.

    Behavior:
      - Per-paper dedupe via the version-aware normalized key (handled by
        `normalize_paper_key`), so legacy `vN` chunks and current chunks of the
        same paper collapse to a single citation entry.
      - Under low-confidence mode, individual chunks must clear the confidence bar
        themselves to be cited. This prevents the trust-undermining UX where the
        answer disclaims weak evidence yet the UI still shows several
        unrelated-looking source chips. If no chunk qualifies, citations are empty.
    """
    settings = get_settings()
    low_confidence = bool(state.get("low_confidence"))
    floor = settings.retrieval_low_confidence_threshold if low_confidence else None

    citations: List[Dict] = []
    seen: set[str] = set()
    for idx, chunk in enumerate(state.get("context_chunks", []) or [], start=1):
        if floor is not None:
            score = float(chunk.get("rerank_score", chunk.get("score", 0.0)) or 0.0)
            if score < floor:
                continue
        meta = chunk.get("metadata") or {}
        key = normalize_paper_key(meta)
        if not key or key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "n": idx,
                "title": meta.get("title"),
                "source": meta.get("source"),
                "paper_id": meta.get("paper_id"),
                "arxiv_id": meta.get("arxiv_id"),
                "year": meta.get("year"),
                "authors": meta.get("authors") or [],
            }
        )
    state["citations"] = citations
    return state


def _format_history(history: List[Dict[str, str]] | None) -> str:
    if not history:
        return "(no prior turns)"
    # Drop the most recent user turn — it is already passed as `query`.
    trimmed = list(history)
    if trimmed and trimmed[-1].get("role") == "user":
        trimmed = trimmed[:-1]
    if not trimmed:
        return "(no prior turns)"
    trimmed = trimmed[-HISTORY_TURN_BUDGET:]

    lines: List[str] = []
    used = 0
    for turn in trimmed:
        role = (turn.get("role") or "user").lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        prefix = "User" if role == "user" else "Assistant"
        line = f"{prefix}: {content}"
        if used + len(line) > HISTORY_CHAR_BUDGET:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines) if lines else "(no prior turns)"


def summarization_agent(state: AgentState) -> AgentState:
    chunks = state.get("context_chunks") or []
    if not chunks:
        state["context"] = ""
        state["response"] = (
            "I couldn't retrieve any passages relevant to that question. "
            "If you uploaded a PDF, confirm it indexed successfully (try re-uploading). "
            "Otherwise try rephrasing — broader topical terms usually help."
        )
        state["citations"] = []
        return state

    settings = get_settings()
    state["context"] = build_context(chunks)
    history_block = _format_history(state.get("history"))

    # Confidence flag was already set in retrieval_agent. We just read it here so the
    # system prompt can be tightened and the UI banner stays consistent.
    low_confidence = bool(state.get("low_confidence"))

    system_prompt = (
        "You are a rigorous research assistant. Never invent citations or papers outside "
        "the user prompt's Context. Always include inline [n] citation markers that map "
        "to the numbered Context blocks."
    )
    if low_confidence:
        system_prompt += (
            " The retrieved evidence has LOW confidence for this query. Begin your reply with "
            "the literal sentence 'Note: retrieved evidence is weak for this question.' on its own line, "
            "then answer ONLY what the Context truly supports — do not extrapolate or speculate. "
            "If the Context does not actually answer the question, say so explicitly."
        )

    prompt = RETRIEVAL_QA_PROMPT.format(
        query=state["query"],
        context=state["context"],
        history=history_block,
    )

    client = OpenAI(api_key=settings.openai_api_key)
    t0 = time.monotonic()
    completion = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    llm_ms = int((time.monotonic() - t0) * 1000)
    answer = completion.choices[0].message.content or ""
    logger.info("Generation timing: llm_ms=%s", llm_ms)
    state["response"] = answer

    # Disclaimer-aware citation suppression. If the LLM itself said the corpus doesn't
    # answer the question, the citation chips below it are misleading even when the
    # rerank scores were high (corpus shares vocabulary but not actual content).
    if _answer_is_disclaimer(answer):
        state["citations"] = []
        state["low_confidence"] = True
    return state


def build_research_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("citation_agent", citation_agent)
    graph.add_node("summarization_agent", summarization_agent)

    graph.set_entry_point("retrieval_agent")
    graph.add_edge("retrieval_agent", "citation_agent")
    graph.add_edge("citation_agent", "summarization_agent")
    graph.add_edge("summarization_agent", END)
    return graph.compile()
