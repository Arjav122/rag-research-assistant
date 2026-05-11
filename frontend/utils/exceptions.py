"""Frontend exceptions (standalone module avoids stale api_client import issues in Docker)."""

from typing import Optional


class APIError(Exception):
    """Raised when backend HTTP calls fail."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
