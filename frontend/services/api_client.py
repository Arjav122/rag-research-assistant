import os

import requests

from frontend.utils.exceptions import APIError
from frontend.services.upload_api import upload_pdf

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


def post(path: str, payload: dict, timeout: int = 300) -> dict:
    url = f"{BACKEND_URL.rstrip('/')}{path}"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
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


def get(path: str, timeout: int = 60) -> dict:
    url = f"{BACKEND_URL.rstrip('/')}{path}"
    try:
        response = requests.get(url, timeout=timeout)
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


__all__ = ["APIError", "post", "get", "upload_pdf"]
