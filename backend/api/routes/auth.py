from fastapi import APIRouter

from backend.api.schemas import APIResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/health", response_model=APIResponse)
def auth_health():
    return APIResponse(success=True, message="Auth scaffold ready", data={"status": "todo"})
