from fastapi import APIRouter, HTTPException

from backend.api.schemas import ComparisonRequest, APIResponse
from backend.services.comparison_service import compare_papers

router = APIRouter(prefix="/comparison", tags=["comparison"])


@router.post("/", response_model=APIResponse)
def compare(payload: ComparisonRequest):
    try:
        data = compare_papers(payload.paper_ids)
        return APIResponse(success=True, message="Comparison generated", data=data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
