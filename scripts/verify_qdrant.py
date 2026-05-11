"""CLI: Qdrant collection stats (point count, vector size, sample payloads)."""

import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

from backend.db.qdrant_client import get_qdrant_client
from backend.ingestion.verification import ingestion_verification_report


if __name__ == "__main__":
    client = get_qdrant_client()
    report = ingestion_verification_report(client)
    print(json.dumps(report, indent=2))
