"""
StoryWeave AI - AI Intelligence API

Endpoints for interactive story universe queries (RAG chat), content generation,
and continuity checking.
"""

from fastapi import APIRouter, Depends, status
from app.core.security import get_current_user_id
from app.services.rag_service import RAGService, get_rag_service
from app.services.writing_assistant_service import (
    WritingAssistantService,
    get_writing_assistant_service,
)
from app.schemas.rag import ChatRequest, ChatResponse
from app.schemas.assistant import AssistantChatRequest, AssistantChatResponse

router = APIRouter(tags=["AI Intelligence"])


@router.post(
    "/ai/chat",
    response_model=ChatResponse,
    summary="Query story universe with RAG",
    description=(
        "Retrieval-Augmented Generation (RAG) chat endpoint. "
        "Retrieves semantically relevant manuscript memories for the specified story universe, "
        "builds a grounded context, and generates an answer using Google Gemini Flash. "
        "Returns the answer along with source chapter citations, relevance metrics, and "
        "uncertainty flags if context is insufficient."
    ),
)
async def ai_chat(
    request: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    """
    Execute grounded RAG chat query.

    NOTE: Authentication is currently using development identity via get_current_user_id.
    Arbitrary client-supplied user_id parameters are not accepted; ownership is enforced
    via the authenticated security dependency.
    """
    return await rag_service.answer_question(
        story_id=request.story_id,
        message=request.message,
        user_id=user_id,
        top_k=request.top_k or 5,
        min_relevance=request.min_relevance if request.min_relevance is not None else 0.25,
    )


@router.post(
    "/ai/assistant",
    response_model=AssistantChatResponse,
    summary="Story-aware writing assistant",
    description=(
        "Consultative writing advisor that evaluates character consistency, plot tension, "
        "dialogue voice, pacing, and continuity against established story canon. "
        "Offers optional creative suggestions while strictly respecting author ownership."
    ),
)
async def writing_assistant(
    request: AssistantChatRequest,
    user_id: str = Depends(get_current_user_id),
    assistant_service: WritingAssistantService = Depends(get_writing_assistant_service),
) -> AssistantChatResponse:
    return await assistant_service.provide_feedback(request=request, user_id=user_id)


@router.post("/ai/generate", summary="AI content generation", status_code=501)
async def ai_generate():
    """Generate story content with AI co-writer. (Phase 6)"""
    return {"message": "Not implemented yet — coming in Phase 6"}


@router.post(
    "/ai/consistency-check",
    summary="Continuity consistency check",
    status_code=501,
)
async def consistency_check():
    """Check story consistency and detect contradictions. (Phase 6)"""
    return {"message": "Not implemented yet — coming in Phase 6"}


@router.post("/ai/plot-analysis", summary="Plot analysis", status_code=501)
async def plot_analysis():
    """Analyze plot structure and detect potential issues. (Phase 6)"""
    return {"message": "Not implemented yet — coming in Phase 6"}
