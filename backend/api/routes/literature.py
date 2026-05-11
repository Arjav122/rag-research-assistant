from fastapi import APIRouter, HTTPException

from backend.api.schemas import LiteratureReviewRequest, APIResponse
from backend.services.literature_service import generate_literature_review

router = APIRouter(prefix="/literature", tags=["literature"])


@router.post("/review", response_model=APIResponse)
def literature_review(payload: LiteratureReviewRequest):
    try:
        data = generate_literature_review(payload.topic, payload.max_papers)
        return APIResponse(success=True, message="Literature review generated", data=data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
