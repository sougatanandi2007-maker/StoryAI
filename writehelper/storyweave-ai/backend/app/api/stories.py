"""
StoryWeave AI - Stories API

Full CRUD operations for fictional universes.
"""

from fastapi import APIRouter, Depends, Query, status
from app.core.security import get_current_user_id
from app.services.story_service import StoryService, get_story_service
from app.schemas.story import (
    StoryCreate,
    StoryUpdate,
    StoryResponse,
    PaginatedStoryResponse,
)

router = APIRouter(tags=["Stories"])


@router.post(
    "/stories",
    response_model=StoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new fictional universe",
    description="Initialize a new story universe with world rules, premise, and metadata.",
)
async def create_story(
    story_in: StoryCreate,
    user_id: str = Depends(get_current_user_id),
    story_service: StoryService = Depends(get_story_service),
) -> StoryResponse:
    return await story_service.create_story(user_id=user_id, story_in=story_in)


@router.get(
    "/stories",
    response_model=PaginatedStoryResponse,
    summary="List all story universes",
    description="Retrieve a paginated list of stories owned by the current user.",
)
async def list_stories(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
    user_id: str = Depends(get_current_user_id),
    story_service: StoryService = Depends(get_story_service),
) -> PaginatedStoryResponse:
    return await story_service.list_stories(user_id=user_id, page=page, limit=limit)


@router.get(
    "/stories/{story_id}",
    response_model=StoryResponse,
    summary="Get story details",
    description="Retrieve complete story metadata and chapter count by story ID.",
)
async def get_story(
    story_id: str,
    user_id: str = Depends(get_current_user_id),
    story_service: StoryService = Depends(get_story_service),
) -> StoryResponse:
    return await story_service.get_story(story_id=story_id, user_id=user_id)


@router.put(
    "/stories/{story_id}",
    response_model=StoryResponse,
    summary="Update a story",
    description="Update title, synopsis, genre, world rules, or metadata for a story.",
)
async def update_story(
    story_id: str,
    story_in: StoryUpdate,
    user_id: str = Depends(get_current_user_id),
    story_service: StoryService = Depends(get_story_service),
) -> StoryResponse:
    return await story_service.update_story(
        story_id=story_id, user_id=user_id, story_in=story_in
    )


@router.delete(
    "/stories/{story_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a story",
    description="Safely delete a story and all chapters belonging to it.",
)
async def delete_story(
    story_id: str,
    user_id: str = Depends(get_current_user_id),
    story_service: StoryService = Depends(get_story_service),
) -> None:
    await story_service.delete_story(story_id=story_id, user_id=user_id)
