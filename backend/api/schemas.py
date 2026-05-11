from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    # Default targets ~300 paper coverage; increase further once pipeline + retrieval are validated.
    max_results: int = Field(default=300, ge=1, le=5000)
    topics: List[str] = Field(
        default_factory=lambda: [
            # Core LLM / NLP foundation
            "Large Language Models",
            "Retrieval Augmented Generation",
            "Natural Language Processing",
            # Alignment / fine-tuning
            "RLHF",
            "Instruction Tuning",
            "Alignment",
            # Reasoning / agents
            "Chain of Thought Reasoning",
            "AI Agents",
            "Tool Use",
            # Multimodal / vision
            "Multimodal Learning",
            "Vision Language Models",
            "Diffusion Models",
            # Evaluation / robustness
            "LLM Evaluation",
            "Benchmarks",
            # Efficiency
            "Mixture of Experts",
            "Long Context",
        ]
    )


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class ChatRequest(BaseModel):
    query: str
    session_id: str
    top_k: int = 8
    # Scope retrieval so answers come from uploads only or one uploaded paper (see docs).
    retrieval_scope: Literal["all", "user_uploads"] = "all"
    restrict_to_paper_id: Optional[str] = Field(
        default=None,
        description='Optional exact payload paper_id, e.g. "user:<uuid>" after PDF upload.',
    )


class LiteratureReviewRequest(BaseModel):
    topic: str
    max_papers: int = 20


class ComparisonRequest(BaseModel):
    paper_ids: List[str]


class RecommendationRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    top_k: int = 10


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None
