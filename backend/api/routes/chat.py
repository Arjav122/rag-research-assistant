from fastapi import APIRouter, HTTPException

from backend.api.schemas import ChatRequest, APIResponse
from backend.services.chat_service import chat_with_research_assistant, reset_session

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=APIResponse)
def chat(payload: ChatRequest):
    try:
        data = chat_with_research_assistant(
            query=payload.query,
            session_id=payload.session_id,
            top_k=payload.top_k,
            retrieval_scope=payload.retrieval_scope,
            restrict_to_paper_id=payload.restrict_to_paper_id,
        )
        return APIResponse(success=True, message="Chat response generated", data=data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/reset", response_model=APIResponse)
def chat_reset(session_id: str):
    """Clear server-side conversation memory for a session."""
    try:
        reset_session(session_id)
        return APIResponse(success=True, message="Session memory cleared", data={"session_id": session_id})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
