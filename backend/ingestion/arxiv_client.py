"""Robust arXiv Atom API client: pagination, feedparser, rich metadata."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import feedparser
import httpx

from backend.utils.config import get_settings

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"

DEFAULT_USER_AGENT = (
    "AI-Research-Assistant/1.0 (ingestion; respectful use; +https://info.arxiv.org/help/api/index.html)"
)


def _quote_topic(topic: str) -> str:
    t = topic.strip()
    if not t:
        return ""
    if " " in t:
        return f'all:"{t}"'
    return f"all:{t}"


def build_arxiv_query(topics: List[str]) -> str:
    parts = [_quote_topic(t) for t in topics if t.strip()]
    if not parts:
        raise ValueError("At least one non-empty topic is required")
    return "(" + " OR ".join(parts) + ")"


def _opensearch_total_from_parsed(parsed: Any) -> int | None:
    if not getattr(parsed, "feed", None):
        return None
    f = parsed.feed
    for key in ("opensearch_totalresults", "opensearch_totalResults"):
        v = getattr(f, key, None)
        if v is None and isinstance(f, dict):
            v = f.get(key)
        if v is not None:
            try:
                return int(str(v).strip())
            except (TypeError, ValueError):
                pass
    return None


def _opensearch_total_from_xml(xml_text: str) -> int | None:
    m = re.search(
        r"<opensearch:totalResults[^>]*>(\d+)</opensearch:totalResults>",
        xml_text,
        re.I,
    )
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _arxiv_id_from_entry_id(entry_id: str) -> str:
    return entry_id.rstrip("/").split("/")[-1]


def _arxiv_base_id(arxiv_id: str) -> str:
    if not arxiv_id:
        return ""
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def _year_from_published(published: str | None) -> int | None:
    if not published:
        return None
    try:
        return int(published[:4])
    except (TypeError, ValueError):
        return None


def _normalize_entry(entry: Any) -> Dict[str, Any] | None:
    entry_id = getattr(entry, "id", None) or ""
    if not entry_id:
        return None

    arxiv_id = _arxiv_id_from_entry_id(entry_id)
    title = (getattr(entry, "title", "") or "").replace("\n", " ").strip()
    summary = (getattr(entry, "summary", "") or "").strip()

    authors: List[str] = []
    for a in getattr(entry, "authors", []) or []:
        name = a.get("name") if isinstance(a, dict) else getattr(a, "name", None)
        if name:
            authors.append(str(name).strip())

    pdf_url = ""
    for link in getattr(entry, "links", []) or []:
        href = (link.get("href") or "").strip()
        if not href:
            continue
        if link.get("type") == "application/pdf" or link.get("title") == "pdf":
            pdf_url = href.replace("http://", "https://", 1)
            break

    primary = None
    apc = getattr(entry, "arxiv_primary_category", None)
    if apc is not None:
        primary = getattr(apc, "term", None) or (apc.get("term") if isinstance(apc, dict) else None)

    categories: List[str] = []
    for tag in getattr(entry, "tags", []) or []:
        term = getattr(tag, "term", None) or (tag.get("term") if isinstance(tag, dict) else None)
        if term:
            categories.append(str(term))

    published = getattr(entry, "published", None) or getattr(entry, "updated", None)

    base_id = _arxiv_base_id(arxiv_id)
    return {
        "paper_id": f"arxiv:{base_id or arxiv_id}",
        "arxiv_id": arxiv_id,
        "arxiv_base_id": base_id,
        "abs_url": entry_id.replace("http://", "https://", 1) if entry_id else "",
        "title": title,
        "summary": summary,
        "authors": authors,
        "year": _year_from_published(published),
        "published": published or "",
        "primary_category": primary,
        "arxiv_categories": categories,
        "pdf_url": pdf_url,
        "source": "arxiv",
    }


async def _fetch_page(
    client: httpx.AsyncClient,
    query: str,
    start: int,
    page_size: int,
    *,
    sort_descending: bool = True,
) -> tuple[List[Dict[str, Any]], int, int | None]:
    """
    Returns (papers_with_pdf, raw_entry_count_from_api, opensearch_total_results_or_none).
    """
    params = {
        "search_query": query,
        "start": start,
        "max_results": page_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending" if sort_descending else "ascending",
    }
    xml_text = ""
    response = None
    for attempt in range(8):
        response = await client.get(ARXIV_API, params=params)
        if response.status_code == 429:
            wait_s = min(180.0, 12.0 * (2**attempt))
            logger.warning(
                "arXiv returned 429 rate limit at start=%s; sleeping %.0fs (attempt %s/8)",
                start,
                wait_s,
                attempt + 1,
            )
            await asyncio.sleep(wait_s)
            continue
        if response.status_code in (502, 503, 504):
            wait_s = min(120.0, 8.0 * (2**attempt))
            logger.warning(
                "arXiv returned %s at start=%s; sleeping %.0fs (attempt %s/8)",
                response.status_code,
                start,
                wait_s,
                attempt + 1,
            )
            await asyncio.sleep(wait_s)
            continue
        response.raise_for_status()
        xml_text = response.text
        break
    else:
        raise RuntimeError(
            "arXiv API failed after 8 attempts (retryable codes only: 429/502/503/504). "
            "Wait several minutes and run ingestion again."
        )

    def _parse() -> tuple[List[Dict[str, Any]], int, int | None]:
        parsed = feedparser.parse(xml_text)
        if getattr(parsed, "bozo", False) and not parsed.entries:
            logger.warning("feedparser reported parse issues: %s", getattr(parsed, "bozo_exception", None))
        total = _opensearch_total_from_parsed(parsed)
        if total is None:
            total = _opensearch_total_from_xml(xml_text)
        raw_entries = list(parsed.entries)
        out: List[Dict[str, Any]] = []
        for entry in raw_entries:
            norm = _normalize_entry(entry)
            if norm and norm.get("pdf_url"):
                out.append(norm)
            elif norm:
                logger.debug("Skipping entry without PDF link: %s", norm.get("arxiv_id"))
        return out, len(raw_entries), total

    return await asyncio.to_thread(_parse)


async def _collect_papers_for_query(
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
    *,
    requested_start_offset: int,
    page_size: int,
    sort_descending: bool,
    delay: float,
    empty_page_retries: int = 4,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Probe start=0 for totalResults, clamp offset, then collect up to max_results PDF papers.
    """
    meta: Dict[str, Any] = {
        "effective_query": query,
        "requested_start_offset": requested_start_offset,
        "effective_start_offset": 0,
        "estimated_total_results": None,
        "offset_clamped_to_zero": False,
    }

    page0, raw0, total_est = await _fetch_page(client, query, 0, page_size, sort_descending=sort_descending)
    meta["estimated_total_results"] = total_est

    eff_start = max(0, int(requested_start_offset))
    if total_est is not None and eff_start >= total_est:
        logger.info(
            "arXiv offset %s >= estimated total results %s; using start=0",
            eff_start,
            total_est,
        )
        eff_start = 0
        meta["offset_clamped_to_zero"] = True

    meta["effective_start_offset"] = eff_start

    collected: List[Dict[str, Any]] = []
    seen: set[str] = set()

    if eff_start == 0:
        for paper in page0:
            aid = paper.get("arxiv_base_id") or paper["arxiv_id"]
            if aid in seen:
                continue
            seen.add(aid)
            collected.append(paper)
            if len(collected) >= max_results:
                return collected[:max_results], meta
        start = raw0
        empty_first_page_retries = 0
    else:
        page, raw_returned, _ = await _fetch_page(
            client, query, eff_start, page_size, sort_descending=sort_descending
        )
        if raw_returned == 0:
            logger.info("No arXiv entries at clamped offset %s; stopping", eff_start)
            return [], meta
        for paper in page:
            aid = paper.get("arxiv_base_id") or paper["arxiv_id"]
            if aid in seen:
                continue
            seen.add(aid)
            collected.append(paper)
            if len(collected) >= max_results:
                return collected[:max_results], meta
        start = eff_start + raw_returned
        empty_first_page_retries = 0

    while len(collected) < max_results:
        page, raw_returned, _ = await _fetch_page(client, query, start, page_size, sort_descending=sort_descending)
        if raw_returned == 0:
            if start == 0 and len(collected) == 0 and empty_first_page_retries < empty_page_retries:
                empty_first_page_retries += 1
                wait_empty = 90.0 * empty_first_page_retries
                logger.warning(
                    "arXiv returned zero entries at start=0 (likely post-429 flake); "
                    "sleeping %.0fs then retry (%s/%s)",
                    wait_empty,
                    empty_first_page_retries,
                    empty_page_retries,
                )
                await asyncio.sleep(wait_empty)
                continue
            logger.info("No more arXiv entries at start=%s; stopping pagination", start)
            break

        for paper in page:
            aid = paper.get("arxiv_base_id") or paper["arxiv_id"]
            if aid in seen:
                continue
            seen.add(aid)
            collected.append(paper)
            if len(collected) >= max_results:
                return collected[:max_results], meta

        start += raw_returned
        if raw_returned < page_size:
            break
        if len(collected) < max_results:
            await asyncio.sleep(delay)

    return collected[:max_results], meta


async def fetch_arxiv_papers(
    topics: List[str],
    max_results: int,
    *,
    page_size: int | None = None,
    inter_request_delay_seconds: float | None = None,
    start_offset: int = 0,
    sort_descending: bool = True,
    raw_query: str | None = None,
    fallback_raw_query: str | None = None,
    fetch_report: dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Fetch papers from arXiv. Use either `raw_query` (boolean-style arXiv query) or `topics`
    (legacy all:"phrase" builder). Optional `fallback_raw_query` tries once if the primary returns 0.
    """
    settings = get_settings()
    page_size = min(page_size or settings.arxiv_page_size, 30000)
    delay = (
        inter_request_delay_seconds
        if inter_request_delay_seconds is not None
        else settings.arxiv_inter_request_delay_seconds
    )

    if raw_query and raw_query.strip():
        primary_q = raw_query.strip()
    else:
        primary_q = build_arxiv_query(topics)

    report = fetch_report if fetch_report is not None else {}
    report["original_query"] = primary_q
    report["relaxed_query"] = None
    report["estimated_total_results"] = None
    report["effective_query"] = primary_q
    report["papers_fetched"] = 0

    logger.info(
        "arXiv primary query: %s (max_results=%s, page_size=%s, start_offset=%s, sort=%s)",
        primary_q,
        max_results,
        page_size,
        start_offset,
        "desc" if sort_descending else "asc",
    )

    headers = {"User-Agent": DEFAULT_USER_AGENT}

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), headers=headers, follow_redirects=True) as client:
        collected, meta = await _collect_papers_for_query(
            client,
            primary_q,
            max_results,
            requested_start_offset=start_offset,
            page_size=page_size,
            sort_descending=sort_descending,
            delay=delay,
        )
        report.update(meta)
        report["estimated_total_results"] = meta.get("estimated_total_results")

        if not collected and fallback_raw_query and fallback_raw_query.strip():
            fb = fallback_raw_query.strip()
            report["relaxed_query"] = fb
            report["effective_query"] = fb
            logger.info("arXiv adaptive retry with relaxed query: %s", fb)
            collected, meta_fb = await _collect_papers_for_query(
                client,
                fb,
                max_results,
                requested_start_offset=0,
                page_size=page_size,
                sort_descending=sort_descending,
                delay=delay,
            )
            report["estimated_total_results"] = meta_fb.get("estimated_total_results")
            report["effective_start_offset"] = meta_fb.get("effective_start_offset", 0)

    report["papers_fetched"] = len(collected)
    logger.info(
        "Fetched %s unique papers with PDF (effective_query=%s, est_total=%s)",
        len(collected),
        report.get("effective_query"),
        report.get("estimated_total_results"),
    )
    return collected
