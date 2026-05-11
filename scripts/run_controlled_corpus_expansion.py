"""Quality-focused corpus expansion toward ~500 unique papers (curated AI/ML).

* Broad OR queries + category filter (cs.AI, cs.LG, cs.CL, cs.CV, cs.IR, stat.ML)
* Recent papers only: sort descending (newest first) for every slice
* Stops gracefully when unique paper count >= --corpus-cap (default 500)
* Adaptive fallback query, skip-if-indexed, totalResults clamp, start_offset=0
* Duplicate-saturated slices: if offset=0 returns only already-indexed papers, one controlled
  retry at a small offset step (see SATURATION_OFFSET_STEP)

Docker:
  docker compose exec -T backend python scripts/run_controlled_corpus_expansion.py
  docker compose exec -T backend python scripts/run_controlled_corpus_expansion.py --corpus-cap 500 --fast --per-slice 8
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from backend.db.qdrant_client import get_qdrant_client
from backend.db.qdrant_indexed_papers import summarize_corpus
from backend.ingestion.pipeline import ingest_arxiv_pipeline
from backend.utils.config import get_settings

_CAT = "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:cs.IR OR cat:stat.ML)"

# Slices ordered with *coverage-first* themes (embeddings, reranking, query understanding, …)
# before broader RAG/LLM slices to reduce duplicate-heavy overlap with already-indexed hot queries.
BROAD_SLICES: list[dict[str, str]] = [
    {
        "primary": '(abs:"text embedding" OR abs:"sentence embedding" OR abs:"embedding model" OR abs:"contrastive learning" OR abs:"dense encoder" OR abs:"sentence transformer")',
        "fallback": '(all:embedding AND (all:retrieval OR all:passage OR all:sentence))',
    },
    {
        "primary": '(abs:rerank OR abs:"cross-encoder" OR abs:"re-ranking" OR abs:"reranking model" OR abs:listwise OR abs:monoT5)',
        "fallback": '(all:rerank OR all:"cross encoder" OR all:ranking)',
    },
    {
        "primary": '(abs:"query expansion" OR abs:"query rewriting" OR abs:"query reformulation" OR abs:HyDE OR abs:"pseudo relevance")',
        "fallback": '(all:"query expansion" OR all:"query rewriting" OR all:retrieval)',
    },
    {
        "primary": '(abs:"passage ranking" OR abs:"learning to rank" OR abs:"neural IR" OR (abs:ranking AND abs:retrieval))',
        "fallback": '(all:"passage retrieval" OR all:"learning to rank")',
    },
    {
        "primary": '(abs:"faithful generation" OR abs:"grounded generation" OR abs:"attributed generation" OR (abs:factuality AND abs:generation))',
        "fallback": '(all:faithful OR all:grounded OR all:"retrieval augmented")',
    },
    {
        "primary": '(abs:"hallucination detection" OR abs:"hallucination mitigation" OR abs:"reduce hallucination" OR abs:"factual consistency")',
        "fallback": '(all:hallucination OR all:factual OR all:grounding)',
    },
    {
        "primary": '(abs:"function calling" OR abs:"tool calling" OR abs:"tool augmented" OR abs:"API augmented" OR abs:"software tools")',
        "fallback": '(all:"tool use" OR all:"function calling" OR all:agent)',
    },
    {
        "primary": '((abs:"instruction following" AND abs:retrieval) OR (abs:"instruction tuning" AND abs:document))',
        "fallback": '(all:instruction AND (all:retrieval OR all:RAG))',
    },
    {
        "primary": '(abs:"LLM-as-judge" OR abs:"automated evaluation" OR abs:"evaluation benchmark" OR (abs:evaluation AND abs:"large language model"))',
        "fallback": '(all:benchmark AND all:evaluation)',
    },
    {
        "primary": '(abs:SPLADE OR abs:"sparse retrieval" OR abs:"hybrid retrieval" OR abs:"lexical semantic" OR (abs:BM25 AND abs:neural))',
        "fallback": '(all:hybrid OR all:sparse OR all:BM25)',
    },
    {
        "primary": '((abs:"open-domain" AND abs:question) OR (abs:"reading comprehension" AND abs:retrieval))',
        "fallback": '(all:"question answering" AND all:retrieval)',
    },
    {
        "primary": '(abs:"semantic cache" OR abs:"key-value memory" OR abs:"compressed memory" OR (abs:"long-term context" AND abs:language))',
        "fallback": '(all:memory AND all:context)',
    },
    {
        "primary": '(abs:"retrieval augmented" OR abs:RAG OR ti:"knowledge graph" OR all:GraphRAG OR abs:"graph retrieval")',
        "fallback": '(all:"retrieval augmented generation" OR all:GraphRAG OR all:RAG)',
    },
    {
        "primary": '(ti:"graph neural" OR abs:"graph neural network" OR abs:"knowledge graph" OR ti:"graph reasoning")',
        "fallback": '(all:"graph neural network" OR all:"knowledge graph")',
    },
    {
        "primary": '(abs:"multi-hop" OR abs:"multi hop" OR abs:"iterative retrieval" OR ti:"chain of reasoning")',
        "fallback": '(all:"multi-hop reasoning" OR all:"iterative retrieval")',
    },
    {
        "primary": '(abs:"chain of thought" OR abs:"chain-of-thought" OR (abs:reasoning AND abs:retrieval))',
        "fallback": '(all:"chain of thought" OR abs:reasoning)',
    },
    {
        "primary": '(abs:"external memory" OR abs:"neural memory" OR abs:"memory augmented" OR (ti:memory AND abs:transformer))',
        "fallback": '(all:"memory augmented" OR all:"external memory" OR all:transformer)',
    },
    {
        "primary": '(abs:"long context" OR abs:"context window" OR abs:"context length" OR (abs:extrapolation AND abs:"language model"))',
        "fallback": '(all:"long context" OR all:"context window" OR all:transformer)',
    },
    {
        "primary": '(abs:"tool use" OR abs:"tool-use" OR abs:"software agents" OR abs:"autonomous agent" OR abs:"LLM agent")',
        "fallback": '(all:"language model" AND all:agent)',
    },
    {
        "primary": '(abs:"orchestrat" OR abs:"workflow" OR abs:"multi-agent" OR abs:"agentic")',
        "fallback": '(all:agent AND all:workflow)',
    },
    {
        "primary": '(abs:"multimodal" OR abs:"vision-language" OR abs:"vision language" OR ti:VLM)',
        "fallback": '(all:"vision language" OR all:multimodal)',
    },
    {
        "primary": '(abs:"grounding" OR abs:"hallucination" OR abs:"factual" OR abs:"attribution" OR abs:citation)',
        "fallback": '(all:grounding OR all:hallucination OR all:RAG)',
    },
    {
        "primary": '(abs:"benchmark" OR abs:"evaluation" OR abs:"RAG evaluation" OR (abs:"question answering" AND abs:dataset))',
        "fallback": '(all:benchmark AND all:"language model")',
    },
    {
        "primary": '(abs:"episodic memory" OR abs:"semantic memory" OR abs:"dialogue memory" OR abs:"session memory")',
        "fallback": '(all:memory AND all:dialogue)',
    },
    {
        "primary": '(abs:"retrieval planning" OR abs:"search plan" OR abs:"strategic retrieval" OR abs:"multi-step retrieval")',
        "fallback": '(all:retrieval AND all:planning)',
    },
    {
        "primary": '(abs:"personalized" OR abs:"contextual memory" OR (abs:"long-term memory" AND abs:dialogue))',
        "fallback": '(all:personalization OR all:dialogue)',
    },
    {
        "primary": '(abs:"evidence" OR abs:"document grounded" OR abs:"retrieval-augmented decision")',
        "fallback": '(all:evidence AND all:retrieval)',
    },
    {
        "primary": '(abs:"dense retrieval" OR abs:"passage retrieval" OR abs:"semantic search" OR abs:BM25)',
        "fallback": '(all:retrieval AND all:passage)',
    },
    {
        "primary": '(abs:"instruction tuning" OR abs:"supervised fine-tuning" OR abs:alignment OR abs:RLHF)',
        "fallback": '(all:RLHF OR all:"instruction tuning")',
    },
    {
        "primary": '(abs:"mixture of experts" OR abs:MoE OR abs:"parameter efficient" OR abs:LoRA)',
        "fallback": '(all:LoRA OR all:MoE)',
    },
    {
        "primary": '((abs:"reinforcement learning" AND abs:language) OR abs:RLHF OR abs:"human feedback")',
        "fallback": '(all:RLHF OR all:"human feedback")',
    },
    {
        "primary": '((abs:"generative" AND abs:"language model") OR abs:LLM OR abs:"large language model")',
        "fallback": '(all:"large language model" OR all:LLM)',
    },
    {
        "primary": '((abs:"computer vision" AND abs:language) OR abs:"image text" OR abs:"visual question")',
        "fallback": '(all:VQA OR all:"vision language")',
    },
    {
        "primary": '(abs:"in-context learning" OR abs:"few-shot" OR abs:prompting OR abs:"prompt engineering")',
        "fallback": '(all:"in context learning" OR all:prompting)',
    },
    # Extra coverage (post–500 push): specific IR/RAG surfaces with less overlap than mega-broad slices.
    {
        "primary": '(abs:ColBERT OR abs:"late interaction" OR abs:"multi-vector retrieval" OR abs:COIL OR abs:MaxSim)',
        "fallback": '(all:ColBERT OR all:"late interaction")',
    },
    {
        "primary": '(abs:"self-RAG" OR abs:"Self-RAG" OR abs:"corrective retrieval" OR abs:FLARE OR ((abs:"iterative retrieval" AND abs:generation)))',
        "fallback": '(all:"self rag" OR all:FLARE OR all:"corrective retrieval")',
    },
    {
        "primary": '(abs:"query decomposition" OR abs:"sub-query" OR abs:"subquery" OR (abs:decomposition AND abs:retrieval))',
        "fallback": '(all:"query decomposition" OR all:subquery)',
    },
    {
        "primary": '((abs:"knowledge distillation" AND abs:retrieval) OR (abs:"teacher student" AND abs:embedding) OR abs:"distill dense")',
        "fallback": '(all:distillation AND all:retrieval)',
    },
    {
        "primary": '(abs:"claim verification" OR abs:"fact verification" OR abs:"evidence retrieval" OR abs:"attribution detection")',
        "fallback": '(all:verification AND all:retrieval)',
    },
    {
        "primary": '(abs:"semantic chunk" OR abs:"document segmentation" OR abs:"chunking strategy" OR ((abs:chunking OR abs:segmentation) AND abs:RAG))',
        "fallback": '(all:chunking AND all:RAG)',
    },
    {
        "primary": '((abs:"cross-modal" AND abs:retrieval) OR abs:"image-text retrieval" OR abs:"visual retrieval" OR (abs:"multimodal" AND abs:retrieval))',
        "fallback": '(all:"cross modal" AND all:retrieval)',
    },
    {
        "primary": '(abs:"search-augmented" OR abs:"retrieve-then-read" OR (abs:"web search" AND abs:"language model") OR abs:"browse and summarize")',
        "fallback": '(all:"search augmented" OR all:"retrieve then read")',
    },
]

PER_SLICE = 4
DEFAULT_CORPUS_CAP = 500
# When a slice at start=0 returns papers but every hit is already indexed, advance once by this
# amount (arXiv `start` index). Keep small (5–10) to avoid aggressive deep crawling.
SATURATION_OFFSET_STEP = 8
# --fast: shorter waits (still respectful); does not change queries or category filters.
FAST_START_PAUSE_SECONDS = 0.0
FAST_SLICE_DELAY_SECONDS = 0.5
FAST_ARXIV_INTER_REQUEST_DELAY_SECONDS = 0.5


def _duplicate_slice_saturated(batch: dict[str, Any]) -> bool:
    """True when every fetched paper was skipped as duplicate (offset=0 window exhausted)."""
    fetched = int(batch.get("papers_fetched") or 0)
    indexed = int(batch.get("papers_indexed_ok") or 0)
    skipped_dup = int(batch.get("papers_skipped_already") or 0)
    return fetched > 0 and indexed == 0 and skipped_dup >= fetched


def _saturation_retry_makes_sense(fetch_meta: dict[str, Any], step: int) -> bool:
    """Skip retry if probing shows total results too small for `step` (would clamp back to 0)."""
    est = fetch_meta.get("estimated_total_results")
    if est is None:
        return True
    try:
        total = int(est)
    except (TypeError, ValueError):
        return True
    return step < total


def _wrap(inner: str) -> str:
    return f"({inner.strip()}) AND {_CAT}"


def _print_final_corpus_report(
    *,
    reason: str,
    corpus_cap: int,
    start_unique: int,
    end_summary: dict[str, Any],
    combined: dict[str, Any] | None,
) -> None:
    log = logging.getLogger(__name__)
    end_u = end_summary["unique_papers"]
    delta_unique = end_u - start_unique
    log.info("========== CORPUS STOP: %s ==========", reason)
    print("\n========== CORPUS REPORT ==========")
    print(f"Stop reason: {reason}")
    print(f"Corpus cap: {corpus_cap}")
    print(f"Unique papers (start of run): {start_unique}")
    print(f"Unique papers (end): {end_u}")
    print(f"Net new unique papers this run: {delta_unique}")
    print(f"Total vectors / points (Qdrant): {end_summary['points_count']}")
    print(f"Chunks verified by scroll: {end_summary['chunks_scrolled']}")
    if combined:
        print(f"Session newly indexed (pipeline counter): {combined.get('papers_indexed_ok', 0)}")
        print(f"Session duplicates skipped: {combined.get('papers_skipped_already', 0)}")
        print(f"Session chunks added: {combined.get('chunks_indexed', 0)}")
        print(f"Session papers fetched from arXiv: {combined.get('papers_fetched', 0)}")
    print("\nTopic / category distribution (chunk counts, top 25):")
    for cat, cnt in end_summary.get("topic_distribution", []):
        print(f"  {cat}: {cnt}")
    print("===================================\n")


def _merge_stats(prev: dict[str, Any] | None, batch: dict[str, Any]) -> dict[str, Any]:
    if prev is None:
        return dict(batch)
    prev_fetch = prev.get("arxiv_fetch_slices") or []
    cur_fetch = batch.get("arxiv_fetch")
    if cur_fetch:
        prev_fetch = prev_fetch + [cur_fetch]
    return {
        "papers_fetched": prev["papers_fetched"] + batch["papers_fetched"],
        "papers_indexed_ok": prev["papers_indexed_ok"] + batch["papers_indexed_ok"],
        "papers_skipped_already": prev.get("papers_skipped_already", 0)
        + batch.get("papers_skipped_already", 0),
        "papers_skipped": prev["papers_skipped"] + batch["papers_skipped"],
        "chunks_indexed": prev["chunks_indexed"] + batch["chunks_indexed"],
        "errors": prev["errors"] + batch["errors"],
        "chunk_stats_per_paper": prev["chunk_stats_per_paper"] + batch["chunk_stats_per_paper"],
        "qdrant_verification": batch.get("qdrant_verification") or prev.get("qdrant_verification"),
        "arxiv_fetch_slices": prev_fetch,
    }


async def _run_controlled_expansion(
    max_slices: int | None,
    corpus_cap: int,
    saturation_offset_step: int,
    *,
    fast: bool = False,
    per_slice: int | None = None,
) -> dict[str, Any]:
    log = logging.getLogger(__name__)
    settings = get_settings()
    delay = float(settings.arxiv_inter_request_delay_seconds)
    if fast:
        start_pause_s = FAST_START_PAUSE_SECONDS
        slice_delay_s = FAST_SLICE_DELAY_SECONDS
        arxiv_fetch_delay_s: float | None = FAST_ARXIV_INTER_REQUEST_DELAY_SECONDS
    else:
        start_pause_s = 15.0
        slice_delay_s = delay
        arxiv_fetch_delay_s = None
    papers_per_slice = int(per_slice) if per_slice is not None else PER_SLICE
    papers_per_slice = max(1, min(papers_per_slice, 25))
    client = get_qdrant_client()
    collection = settings.qdrant_collection

    start_summary = summarize_corpus(client, collection)
    start_unique = start_summary["unique_papers"]

    if start_unique >= corpus_cap:
        _print_final_corpus_report(
            reason="already_at_or_above_corpus_cap",
            corpus_cap=corpus_cap,
            start_unique=start_unique,
            end_summary=start_summary,
            combined=None,
        )
        return {
            "stopped_early": True,
            "reason": "already_at_cap",
            "start_unique_papers": start_unique,
            "end_unique_papers": start_unique,
        }

    slices = BROAD_SLICES[: max_slices if max_slices is not None else len(BROAD_SLICES)]
    n = len(slices)

    step = max(1, int(saturation_offset_step))
    log.info(
        "Controlled expansion: %s slices, %s papers/slice, corpus_cap=%s, saturation_offset_step=%s, "
        "fast=%s start_pause=%ss slice_delay=%ss arxiv_fetch_delay=%s sort=desc (recent only)",
        n,
        papers_per_slice,
        corpus_cap,
        step,
        fast,
        start_pause_s,
        slice_delay_s,
        arxiv_fetch_delay_s,
    )
    await asyncio.sleep(start_pause_s)

    combined: dict[str, Any] | None = None
    stop_reason = "slices_exhausted"
    slices_run = 0

    for i, spec in enumerate(slices):
        cur = summarize_corpus(client, collection)
        if cur["unique_papers"] >= corpus_cap:
            log.info("Corpus cap %s reached (%s papers); stopping before new slice.", corpus_cap, cur["unique_papers"])
            stop_reason = "corpus_cap_reached"
            break

        inner_p = spec["primary"]
        inner_f = spec["fallback"]
        q_primary = _wrap(inner_p)
        q_fallback = _wrap(inner_f)
        log.info(
            "Slice %s/%s: inner_primary=%r inner_fallback=%r sort=desc (recent only)",
            i + 1,
            n,
            inner_p,
            inner_f,
        )
        batch = await ingest_arxiv_pipeline(
            topics=[],
            max_results=papers_per_slice,
            skip_if_indexed=True,
            arxiv_start_offset=0,
            arxiv_sort_descending=True,
            arxiv_raw_query=q_primary,
            arxiv_fallback_raw_query=q_fallback,
            arxiv_inter_request_delay_seconds=arxiv_fetch_delay_s,
        )
        combined = _merge_stats(combined, batch)
        slices_run = i + 1
        fetch_meta = batch.get("arxiv_fetch") or {}
        log.info(
            "Slice %s result: fetched=%s newly_indexed=%s skipped_dup=%s est_total=%s effective_q=%r",
            i + 1,
            batch["papers_fetched"],
            batch["papers_indexed_ok"],
            batch.get("papers_skipped_already", 0),
            fetch_meta.get("estimated_total_results"),
            fetch_meta.get("effective_query", "")[:120],
        )

        # One controlled offset retry only when this slice is duplicate-saturated at offset 0.
        if _duplicate_slice_saturated(batch) and _saturation_retry_makes_sense(fetch_meta, step):
            cur_mid = summarize_corpus(client, collection)
            if cur_mid["unique_papers"] >= corpus_cap:
                log.info(
                    "Corpus cap %s reached before saturation retry; skipping offset advance.",
                    corpus_cap,
                )
            else:
                old_off = int(fetch_meta.get("effective_start_offset") or 0)
                new_off = step
                log.info(
                    "Duplicate saturation detected (slice %s/%s): fetched=%s indexed=0 skipped_dup=%s "
                    "est_total=%s — advancing offset %s -> %s (single retry, sort=desc)",
                    i + 1,
                    n,
                    batch["papers_fetched"],
                    batch.get("papers_skipped_already", 0),
                    fetch_meta.get("estimated_total_results"),
                    old_off,
                    new_off,
                )
                await asyncio.sleep(slice_delay_s)
                retry_batch = await ingest_arxiv_pipeline(
                    topics=[],
                    max_results=papers_per_slice,
                    skip_if_indexed=True,
                    arxiv_start_offset=int(new_off),
                    arxiv_sort_descending=True,
                    arxiv_raw_query=q_primary,
                    arxiv_fallback_raw_query=q_fallback,
                    arxiv_inter_request_delay_seconds=arxiv_fetch_delay_s,
                )
                combined = _merge_stats(combined, retry_batch)
                rmeta = retry_batch.get("arxiv_fetch") or {}
                log.info(
                    "Duplicate saturation retry result (slice %s/%s): fetched=%s newly_indexed=%s "
                    "skipped_dup=%s effective_start_offset=%s offset_clamped=%s",
                    i + 1,
                    n,
                    retry_batch["papers_fetched"],
                    retry_batch["papers_indexed_ok"],
                    retry_batch.get("papers_skipped_already", 0),
                    rmeta.get("effective_start_offset"),
                    rmeta.get("offset_clamped_to_zero"),
                )
        elif _duplicate_slice_saturated(batch) and not _saturation_retry_makes_sense(fetch_meta, step):
            log.info(
                "Duplicate saturation detected (slice %s/%s) but retry skipped: "
                "offset step %s >= est_total %s (would clamp or repeat window)",
                i + 1,
                n,
                step,
                fetch_meta.get("estimated_total_results"),
            )

        cur2 = summarize_corpus(client, collection)
        if cur2["unique_papers"] >= corpus_cap:
            log.info("Corpus cap %s reached after slice %s (%s papers).", corpus_cap, i + 1, cur2["unique_papers"])
            stop_reason = "corpus_cap_reached"
            if i < n - 1:
                pass
            break

        if i < n - 1:
            await asyncio.sleep(slice_delay_s)

    end_summary = summarize_corpus(client, collection)

    if combined is None:
        combined = {
            "papers_fetched": 0,
            "papers_indexed_ok": 0,
            "papers_skipped_already": 0,
            "papers_skipped": [],
            "chunks_indexed": 0,
            "errors": [],
            "chunk_stats_per_paper": [],
        }

    log.info(
        "Controlled expansion finished (%s): slices_run=%s fetched=%s newly_indexed=%s "
        "skipped_already=%s chunks=%s errors=%s",
        stop_reason,
        slices_run,
        combined["papers_fetched"],
        combined["papers_indexed_ok"],
        combined.get("papers_skipped_already", 0),
        combined["chunks_indexed"],
        len(combined["errors"]),
    )

    _print_final_corpus_report(
        reason=stop_reason,
        corpus_cap=corpus_cap,
        start_unique=start_unique,
        end_summary=end_summary,
        combined=combined,
    )

    out = dict(combined)
    out["stop_reason"] = stop_reason
    out["slices_run"] = slices_run
    out["corpus_cap"] = corpus_cap
    out["start_unique_papers"] = start_unique
    out["end_unique_papers"] = end_summary["unique_papers"]
    out["end_points_count"] = end_summary["points_count"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Controlled arXiv corpus expansion (broad queries, ~500 cap).")
    ap.add_argument(
        "--max-slices",
        type=int,
        default=None,
        help="Limit number of query slices (default: all configured slices).",
    )
    ap.add_argument(
        "--corpus-cap",
        type=int,
        default=DEFAULT_CORPUS_CAP,
        help="Stop when unique paper count in Qdrant reaches this (default: 500).",
    )
    ap.add_argument(
        "--saturation-offset-step",
        type=int,
        default=SATURATION_OFFSET_STEP,
        help=(
            "When a slice is duplicate-saturated at offset 0, retry once with this arXiv start index "
            "(small step, typically 5–10; default: 8)."
        ),
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Shorter pauses and lower arXiv inter-request delay (~0.5s) for a quicker run; "
            "same queries, categories, and skip-if-indexed behavior."
        ),
    )
    ap.add_argument(
        "--per-slice",
        type=int,
        default=None,
        help="Papers to fetch per slice (default: 4; max 25). Slightly higher can finish the last few papers faster.",
    )
    args = ap.parse_args()
    print(
        asyncio.run(
            _run_controlled_expansion(
                args.max_slices,
                args.corpus_cap,
                args.saturation_offset_step,
                fast=args.fast,
                per_slice=args.per_slice,
            )
        )
    )


if __name__ == "__main__":
    main()
