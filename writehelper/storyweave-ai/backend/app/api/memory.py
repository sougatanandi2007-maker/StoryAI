"""
StoryWeave AI - Memory API

Semantic story universe memory search, indexing inspection, and status endpoints.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from app.core.security import get_current_user_id
from app.services.memory_service import MemoryService, get_memory_service
from app.schemas.memory import (
    MemorySearchRequest,
    MemorySearchResponse,
    PaginatedMemoryResponse,
    StoryMemoryStatusResponse,
)

router = APIRouter(tags=["Memory"])


@router.post(
    "/memory/search",
    response_model=MemorySearchResponse,
    summary="Search story universe memories",
    description="Perform semantic similarity search over story manuscript memories using vector embeddings.",
)
async def search_memories(
    request: MemorySearchRequest,
    user_id: str = Depends(get_current_user_id),
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemorySearchResponse:
    return await memory_service.search_memories(
        query=request.query,
        user_id=user_id,
        story_id=request.story_id,
        top_k=request.top_k,
        memory_type=request.memory_type,
    )


@router.get(
    "/stories/{story_id}/memories",
    response_model=PaginatedMemoryResponse,
    summary="List stored memories for a story",
    description="Retrieve paginated vector memory chunks for a story universe, optionally filtered by chapter.",
)
async def list_memories(
    story_id: str,
    chapter_id: Optional[str] = Query(default=None, description="Optional chapter ID filter"),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
    user_id: str = Depends(get_current_user_id),
    memory_service: MemoryService = Depends(get_memory_service),
) -> PaginatedMemoryResponse:
    return await memory_service.get_memories(
        story_id=story_id,
        user_id=user_id,
        chapter_id=chapter_id,
        page=page,
        limit=limit,
    )


@router.get(
    "/stories/{story_id}/memory-status",
    response_model=StoryMemoryStatusResponse,
    summary="Get story memory status",
    description="Check memory chunk counts and indexed chapter statistics for a story universe.",
)
async def memory_status(
    story_id: str,
    user_id: str = Depends(get_current_user_id),
    memory_service: MemoryService = Depends(get_memory_service),
) -> StoryMemoryStatusResponse:
    return await memory_service.get_memory_status(
        story_id=story_id, user_id=user_id
    )


@router.delete(
    "/stories/{story_id}/memories",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete all story memories",
    description="Delete all vector memories associated with a story universe.",
)
async def delete_story_memories(
    story_id: str,
    user_id: str = Depends(get_current_user_id),
    memory_service: MemoryService = Depends(get_memory_service),
) -> None:
    await memory_service.delete_story_memories(story_id=story_id, user_id=user_id)
