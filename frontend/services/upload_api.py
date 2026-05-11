"""PDF upload to backend — standalone module (avoids stale docker bind-mount of api_client)."""

import os
from typing import Any, Dict

import requests

from frontend.utils.exceptions import APIError

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


def upload_pdf(file_bytes: bytes, filename: str, timeout: int = 600) -> Dict[str, Any]:
    url = f"{BACKEND_URL.rstrip('/')}/api/v1/ingestion/upload-pdf"
    files = {"file": (filename, file_bytes, "application/pdf")}
    try:
        response = requests.post(url, files=files, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise APIError(str(detail or exc), response.status_code) from exc
    except requests.RequestException as exc:
        raise APIError(f"Network error: {exc}") from exc
