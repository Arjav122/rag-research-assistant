"""PDF download with retries, validation, and PyMuPDF text extraction."""

from __future__ import annotations

import logging
from pathlib import Path

import fitz
import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"
DEFAULT_USER_AGENT = (
    "AI-Research-Assistant/1.0 (PDF download; +https://info.arxiv.org/help/api/index.html)"
)


def _is_pdf_bytes(data: bytes) -> bool:
    if len(data) < 5:
        return False
    return data[:4] == PDF_MAGIC[:4]


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
async def _download_bytes(pdf_url: str, headers: dict[str, str], timeout_seconds: float) -> bytes:
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds),
        headers=headers,
        follow_redirects=True,
    ) as client:
        response = await client.get(pdf_url)
        response.raise_for_status()
        return response.content


async def download_pdf(
    pdf_url: str,
    output_dir: str,
    filename: str,
    *,
    timeout_seconds: float = 120.0,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in filename)[:200]
    file_path = output_path / f"{safe_name}.pdf"

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    data = await _download_bytes(pdf_url, headers, timeout_seconds)
    if not _is_pdf_bytes(data):
        raise ValueError(f"Response is not a PDF (magic check failed): {pdf_url}")
    file_path.write_bytes(data)
    logger.info("Downloaded PDF (%s bytes) -> %s", len(data), file_path)
    return file_path


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text; returns empty string on failure (caller logs)."""
    try:
        with fitz.open(pdf_path) as document:
            parts: list[str] = []
            for page in document:
                parts.append(page.get_text("text") or "")
            return "\n".join(parts).strip()
    except Exception:
        logger.exception("PyMuPDF failed for %s", pdf_path)
        return ""


def extract_text_preview(pdf_path: Path, max_chars: int = 500) -> str:
    text = extract_text_from_pdf(pdf_path)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"
