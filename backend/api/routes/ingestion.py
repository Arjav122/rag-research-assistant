from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.api.schemas import IngestRequest, APIResponse
from backend.db.qdrant_client import get_qdrant_client
from backend.ingestion.pipeline import ingest_arxiv_pipeline
from backend.ingestion.user_upload import ingest_user_pdf_bytes
from backend.ingestion.verification import ingestion_verification_report

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post("/papers", response_model=APIResponse)
async def ingest_papers(payload: IngestRequest):
    try:
        data = await ingest_arxiv_pipeline(topics=payload.topics, max_results=payload.max_results)
        return APIResponse(success=True, message="Ingestion completed", data=data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/upload-pdf", response_model=APIResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Index a user PDF into the same Qdrant collection (chunk → embed → upsert)."""
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a .pdf file")
        raw = await file.read()
        if len(raw) < 100:
            raise HTTPException(status_code=400, detail="File too small")
        data = ingest_user_pdf_bytes(raw, file.filename)
        if not data.get("success"):
            return APIResponse(success=False, message=data.get("error", "upload_failed"), data=data)
        return APIResponse(success=True, message="PDF indexed", data=data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/verify", response_model=APIResponse)
def verify_qdrant_after_ingestion():
    """Qdrant point counts, vector config, and sample payloads for pipeline validation."""
    try:
        client = get_qdrant_client()
        data = ingestion_verification_report(client)
        return APIResponse(success=True, message="Qdrant verification", data=data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
