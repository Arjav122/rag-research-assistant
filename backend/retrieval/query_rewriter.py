"""Query understanding stage: abbreviation expansion, follow-up self-containment, HyDE.

This module is opt-in via settings flags. If the LLM call fails for any reason, we fall
back to the static-expanded query so retrieval still works — never breaks the pipeline.

Output of `prepare_query()`:
  RewriteResult(
    expanded_query: str,            # always set; the query to embed and BM25 against
    hyde_text: Optional[str],       # only set when HyDE is requested AND query is short
    raw_query: str,
    used_llm: bool,
    notes: str,
  )
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI

from backend.utils.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static abbreviation map — covers the high-frequency cases without an LLM call.
# Use Title Case for the expansion so it remains a natural-looking phrase post-substitution.
# ---------------------------------------------------------------------------
STATIC_ABBREVIATIONS: Dict[str, str] = {
    "rag": "Retrieval-Augmented Generation",
    "rlhf": "Reinforcement Learning from Human Feedback",
    "rlaif": "Reinforcement Learning from AI Feedback",
    "dpo": "Direct Preference Optimization",
    "ppo": "Proximal Policy Optimization",
    "sft": "Supervised Fine-Tuning",
    "lora": "Low-Rank Adaptation",
    "qlora": "Quantized Low-Rank Adaptation",
    "moe": "Mixture of Experts",
    "smoe": "Sparse Mixture of Experts",
    "cot": "Chain of Thought",
    "tot": "Tree of Thoughts",
    "ssm": "State Space Model",
    "mamba": "Mamba state space model",
    "vlm": "Vision Language Model",
    "mllm": "Multimodal Large Language Model",
    "vqa": "Visual Question Answering",
    "ner": "Named Entity Recognition",
    "qa": "Question Answering",
    "nlp": "Natural Language Processing",
    "llm": "Large Language Model",
    "llms": "Large Language Models",
    "lm": "Language Model",
    "nli": "Natural Language Inference",
    "ocr": "Optical Character Recognition",
    "asr": "Automatic Speech Recognition",
    "tts": "Text To Speech",
    "gnn": "Graph Neural Network",
    "cnn": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network",
    "lstm": "Long Short-Term Memory",
    "gan": "Generative Adversarial Network",
    "vae": "Variational Autoencoder",
    "kg": "Knowledge Graph",
    "kgrag": "Knowledge Graph Retrieval-Augmented Generation",
    "ann": "Approximate Nearest Neighbor",
    "hnsw": "Hierarchical Navigable Small World",
    "faiss": "Facebook AI Similarity Search",
    "bm25": "BM25 keyword retrieval",
    "tfidf": "TF-IDF",
    "ir": "Information Retrieval",
    "mrr": "Mean Reciprocal Rank",
    "ndcg": "Normalized Discounted Cumulative Gain",
    "mmlu": "Massive Multitask Language Understanding (MMLU benchmark)",
    "gsm8k": "GSM8K math benchmark",
    "humaneval": "HumanEval code benchmark",
    "bbh": "BIG-Bench Hard",
    "arc": "ARC reasoning benchmark",
    "hellaswag": "HellaSwag commonsense benchmark",
    "winogrande": "WinoGrande benchmark",
    "ood": "Out-of-Distribution",
    "ic": "In-Context",
    "icl": "In-Context Learning",
    "fsl": "Few-Shot Learning",
    "zsl": "Zero-Shot Learning",
    "rl": "Reinforcement Learning",
    "rlft": "Reinforcement Learning Fine-Tuning",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "ai": "Artificial Intelligence",
    "agi": "Artificial General Intelligence",
    "asi": "Artificial Super Intelligence",
    "hyde": "Hypothetical Document Embedding",
}


_TOKEN_SPLIT = re.compile(r"([A-Za-z0-9]+)")
_TECHNICAL_TOKEN_RE = re.compile(r"\b(?:[A-Z]{2,}|[A-Za-z]+-[A-Za-z]+|[A-Za-z]+\d+)\b")
_BENCHMARK_TOKEN_RE = re.compile(
    r"\b(?:mmlu|gsm8k|humaneval|bbh|arc|hellaswag|winogrande|truthfulqa|fever)\b",
    re.IGNORECASE,
)


def _expand_static(query: str) -> str:
    """Replace standalone tokens that match the abbreviation map. Preserves spacing."""
    if not query:
        return query
    parts = _TOKEN_SPLIT.split(query)
    out_parts: List[str] = []
    for part in parts:
        if not part:
            out_parts.append(part)
            continue
        key = part.lower()
        expansion = STATIC_ABBREVIATIONS.get(key)
        if expansion:
            # Keep original casing of the abbreviation alongside expansion (helps both
            # dense and sparse retrieval — the dense model gets the expansion, BM25 still
            # matches the literal token if a paper uses it as-is).
            out_parts.append(f"{part} ({expansion})")
        else:
            out_parts.append(part)
    return "".join(out_parts)


# ---------------------------------------------------------------------------
# In-process LRU cache (capped) keyed by raw query + history fingerprint.
# Avoids re-paying LLM cost for repeated turns.
# ---------------------------------------------------------------------------
@dataclass
class RewriteResult:
    expanded_query: str
    hyde_text: Optional[str]
    raw_query: str
    used_llm: bool
    notes: str = ""
    # Detected comparison entities ("X vs Y" pattern). When set, retrieval runs an
    # additional dense pass per entity and RRF-merges the rankings — this fixes the
    # "multi-entity retrieval medium-strength" case where one entity's signal got
    # diluted in a single embedding.
    entities: Optional[List[str]] = None


_COMPARISON_SPLIT_RE = re.compile(
    r"\s+(?:vs\.?|versus|v\.|compared\s+to|against)\s+",
    re.IGNORECASE,
)
_LEADING_INSTRUCTION_RE = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+)?"
    r"(?:compare|contrast|differentiate|explain the difference between|"
    r"what (?:are|is) the difference[s]? between|how does .* differ from|"
    r"what's the difference between)\s*[:\-]?\s*",
    re.IGNORECASE,
)


def detect_comparison_entities(query: str) -> Optional[List[str]]:
    """Return a list of entity strings if the query is shaped like 'X vs Y' (or three+).

    Heuristic-only — no LLM call. Returns None when the pattern doesn't match cleanly,
    so the main retrieval path is unaffected for non-comparison queries.
    """
    if not query:
        return None
    text = query.strip().rstrip("?.! ")
    text = _LEADING_INSTRUCTION_RE.sub("", text).strip()
    if not text:
        return None

    parts = _COMPARISON_SPLIT_RE.split(text)
    if len(parts) < 2:
        return None

    entities: List[str] = []
    for raw_part in parts:
        p = raw_part.strip().strip(",;.").strip()
        # Must have meaningful content: ≥3 chars, ≥1 alphabetic token, not a stopword phrase.
        if len(p) < 3:
            return None
        if not re.search(r"[A-Za-z]", p):
            return None
        # Reject pure-stopword tails like "the others" — must contain at least one
        # token that's either capitalized (proper noun / acronym) or longer than 3 chars.
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", p)
        if not tokens:
            return None
        has_signal = any(
            (t[0:1].isupper() and len(t) > 1) or len(t) > 3 for t in tokens
        )
        if not has_signal:
            return None
        entities.append(p)

    # Cap to first 3 entities to keep retrieval cost bounded.
    return entities[:3] if entities else None


_CACHE: "OrderedDict[str, RewriteResult]" = OrderedDict()


def _cache_get(key: str) -> Optional[RewriteResult]:
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    return None


def _cache_set(key: str, value: RewriteResult, max_size: int) -> None:
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    while len(_CACHE) > max_size:
        _CACHE.popitem(last=False)


def _history_fingerprint(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""
    last_few = history[-4:]
    return "|".join(f"{t.get('role','')}:{(t.get('content') or '')[:120]}" for t in last_few)


def _is_short_query(q: str) -> bool:
    settings = get_settings()
    if not q:
        return True
    return len(q) <= settings.retrieval_hyde_max_chars or len(q.split()) <= settings.retrieval_hyde_max_tokens


def _is_highly_technical_query(q: str) -> bool:
    """Avoid HyDE on terse technical queries where broad semantic expansion hurts precision."""
    if not q:
        return False
    tokens = [t for t in re.findall(r"[A-Za-z0-9\-]+", q) if t]
    if len(tokens) <= 1:
        return False
    technical = sum(1 for t in tokens if _TECHNICAL_TOKEN_RE.search(t))
    ratio = technical / max(len(tokens), 1)
    return technical >= 2 and ratio >= 0.35


def _apply_technical_expansion(q: str) -> str:
    """Add small, precision-oriented synonym tails for technical queries only.

    Strictly controlled:
    - only for highly technical queries
    - max two expansion snippets
    - no generative paraphrasing or long drift-prone tails
    """
    settings = get_settings()
    if not settings.retrieval_use_technical_expansion:
        return q
    if not _is_highly_technical_query(q):
        return q

    low = q.lower()
    additions: List[str] = []

    if "self-rag" in low or "self rag" in low:
        additions.append("self-verification self-correction corrective retrieval")
    if "hallucination" in low and ("detect" in low or "detection" in low):
        additions.append("factual consistency grounding verification")
    if "rerank" in low or "reranking" in low:
        additions.append("cross-encoder reranking retrieval ranking")
    if "grounded generation" in low or ("grounded" in low and "generation" in low):
        additions.append("evidence-grounded generation")
    if "benchmark" in low or _BENCHMARK_TOKEN_RE.search(low):
        additions.append("benchmark evaluation")

    if not additions:
        return q

    # Keep expansion lightweight and bounded.
    uniq: List[str] = []
    for a in additions:
        if a not in uniq:
            uniq.append(a)
        if len(uniq) >= 2:
            break
    return f"{q} | {' | '.join(uniq)}"


def _llm_rewrite(
    raw_query: str,
    statically_expanded: str,
    history: Optional[List[Dict[str, str]]],
    want_hyde: bool,
) -> Optional[Dict[str, str]]:
    """Single LLM call: produces self-contained query + (optionally) hypothetical answer.

    Returns dict with keys: rewritten, hyde (optional). Returns None on any failure.
    """
    settings = get_settings()
    history_block = ""
    if history:
        last = history[-6:]
        history_block = "\n".join(
            f"{(t.get('role') or 'user').upper()}: {(t.get('content') or '').strip()[:400]}"
            for t in last
        )

    instruction_lines = [
        "You rewrite research queries for a retrieval system over an academic corpus.",
        "Goals:",
        "1) Make the query SELF-CONTAINED if the user is referring to prior turns.",
        "2) Expand abbreviations and add likely paraphrases the corpus might use, but stay faithful to the user's intent. Do not invent topics.",
        "3) Keep it 1-2 sentences. Use plain academic language, no marketing words.",
    ]
    if want_hyde:
        instruction_lines.append(
            "4) Also produce a HYPOTHETICAL ANSWER (`hyde`) of 2-3 sentences in the same register as a research paper "
            "abstract. The hypothetical may speculate but must stay on-topic; we will embed it as an additional retrieval probe."
        )

    schema_lines = ['Return STRICT JSON with these keys (no markdown):',
                    '  "rewritten": "<rewritten self-contained query>"']
    if want_hyde:
        schema_lines.append('  "hyde": "<2-3 sentence hypothetical answer>"')

    user_payload = (
        "Conversation so far (may be empty):\n"
        f"{history_block or '(none)'}\n\n"
        f"Raw user query: {raw_query}\n"
        f"Statically expanded form: {statically_expanded}\n\n"
        f"{chr(10).join(schema_lines)}"
    )

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        completion = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": "\n".join(instruction_lines)},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        text = completion.choices[0].message.content or ""
        return json.loads(text)
    except Exception:
        logger.exception("Query rewrite LLM call failed; falling back to static expansion")
        return None


def prepare_query(
    raw_query: str,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    force_no_llm: bool = False,
) -> RewriteResult:
    """Return a `RewriteResult` for use by retrieval. Never raises."""
    settings = get_settings()
    raw = (raw_query or "").strip()
    if not raw:
        return RewriteResult(expanded_query="", hyde_text=None, raw_query="", used_llm=False)

    static_expanded = _expand_static(raw)
    static_expanded = _apply_technical_expansion(static_expanded)
    entities = detect_comparison_entities(raw)

    if force_no_llm or not settings.retrieval_use_query_rewrite:
        return RewriteResult(
            expanded_query=static_expanded,
            hyde_text=None,
            raw_query=raw,
            used_llm=False,
            notes="static_only",
            entities=entities,
        )

    want_hyde = (
        settings.retrieval_use_hyde
        and _is_short_query(raw)
        and not _is_highly_technical_query(raw)
    )
    cache_key = f"hyde={int(want_hyde)}|h={_history_fingerprint(history)}|q={raw}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    llm_out = _llm_rewrite(raw, static_expanded, history, want_hyde)
    if not llm_out:
        result = RewriteResult(
            expanded_query=static_expanded,
            hyde_text=None,
            raw_query=raw,
            used_llm=False,
            notes="llm_failed_static_fallback",
            entities=entities,
        )
        _cache_set(cache_key, result, settings.retrieval_query_rewrite_cache_size)
        return result

    rewritten = (llm_out.get("rewritten") or "").strip() or static_expanded
    # Always carry the static expansion alongside the rewritten form — best of both for
    # acronym recall on BM25 and natural-language similarity on dense retrieval.
    if static_expanded and static_expanded.lower() != raw.lower() and static_expanded.lower() not in rewritten.lower():
        expanded_query = f"{rewritten} | {static_expanded}"
    else:
        expanded_query = rewritten

    hyde_text = (llm_out.get("hyde") or "").strip() if want_hyde else None
    if hyde_text and len(hyde_text) < 30:
        hyde_text = None  # too short to be useful; skip extra retrieval pass

    result = RewriteResult(
        expanded_query=expanded_query,
        hyde_text=hyde_text,
        raw_query=raw,
        used_llm=True,
        notes="ok",
        entities=entities,
    )
    _cache_set(cache_key, result, settings.retrieval_query_rewrite_cache_size)
    return result
