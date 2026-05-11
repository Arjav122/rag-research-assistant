from fastapi import APIRouter, HTTPException

from backend.api.schemas import RecommendationRequest, APIResponse
from backend.services.recommendation_service import recommend_papers

router = APIRouter(prefix="/recommendation", tags=["recommendation"])


@router.post("/", response_model=APIResponse)
def recommendation(payload: RecommendationRequest):
    try:
        data = recommend_papers(payload.query, payload.top_k)
        return APIResponse(success=True, message="Recommendations generated", data=data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
