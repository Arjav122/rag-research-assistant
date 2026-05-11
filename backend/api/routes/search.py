from fastapi import APIRouter, HTTPException

from backend.api.schemas import SearchRequest, APIResponse
from backend.services.search_service import semantic_search

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=APIResponse)
def search(payload: SearchRequest):
    try:
        data = semantic_search(payload.query, payload.top_k)
        return APIResponse(success=True, message="Search completed", data=data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
