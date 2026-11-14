"""
StoryWeave AI - Story Management Service

Encapsulates business logic for fictional universe creation, updates,
retrieval, and safe deletion.
"""

import math
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from app.database.repositories.base import StoryRepository, ChapterRepository
from app.database.repositories.dependencies import get_story_repository, get_chapter_repository
from app.schemas.story import StoryCreate, StoryUpdate, StoryResponse, PaginatedStoryResponse
from app.core.exceptions import StoryNotFoundException
from app.core.logging import get_logger

logger = get_logger("service.story")


class StoryService:
    """Service handling story universe lifecycle."""

    def __init__(self, story_repo: StoryRepository, chapter_repo: ChapterRepository):
        self._story_repo = story_repo
        self._chapter_repo = chapter_repo

    async def create_story(self, user_id: str, story_in: StoryCreate) -> StoryResponse:
        """Create a new fictional universe story."""
        now = datetime.now(timezone.utc)
        story_id = f"story_{uuid.uuid4().hex[:16]}"

        story_dict = {
            "id": story_id,
            "user_id": user_id,
            "title": story_in.title.strip(),
            "synopsis": story_in.synopsis.strip() if story_in.synopsis else None,
            "genre": story_in.genre.strip() if story_in.genre else None,
            "world_rules_overview": (
                story_in.world_rules_overview.strip()
                if story_in.world_rules_overview
                else None
            ),
            "metadata": story_in.metadata or {},
            "created_at": now,
            "updated_at": now,
        }

        created = await self._story_repo.create(story_dict)
        logger.info(f"Created story universe '{created['title']}' (ID: {story_id}) for user '{user_id}'")
        return StoryResponse(**created, chapter_count=0)

    async def get_story(self, story_id: str, user_id: str) -> StoryResponse:
        """Retrieve a specific story by ID with chapter count."""
        story = await self._story_repo.get_by_id(story_id=story_id, user_id=user_id)
        if not story:
            raise StoryNotFoundException(story_id=story_id)

        chapter_count = await self._chapter_repo.count_by_story(story_id=story_id, user_id=user_id)
        return StoryResponse(**story, chapter_count=chapter_count)

    async def list_stories(self, user_id: str, page: int = 1, limit: int = 20) -> PaginatedStoryResponse:
        """List all stories for a user with pagination."""
        skip = (page - 1) * limit
        items_raw, total = await self._story_repo.list(user_id=user_id, skip=skip, limit=limit)

        items: List[StoryResponse] = []
        for story in items_raw:
            ch_count = await self._chapter_repo.count_by_story(story_id=story["id"], user_id=user_id)
            items.append(StoryResponse(**story, chapter_count=ch_count))

        total_pages = math.ceil(total / limit) if total > 0 else 1

        return PaginatedStoryResponse(
            items=items,
            total=total,
            page=page,
            limit=limit,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )

    async def update_story(self, story_id: str, user_id: str, story_in: StoryUpdate) -> StoryResponse:
        """Update story details."""
        existing = await self._story_repo.get_by_id(story_id=story_id, user_id=user_id)
        if not existing:
            raise StoryNotFoundException(story_id=story_id)

        update_dict = story_in.model_dump(exclude_unset=True)
        if "title" in update_dict and update_dict["title"]:
            update_dict["title"] = update_dict["title"].strip()

        updated = await self._story_repo.update(story_id=story_id, user_id=user_id, update_data=update_dict)
        chapter_count = await self._chapter_repo.count_by_story(story_id=story_id, user_id=user_id)
        logger.info(f"Updated story ID: {story_id}")
        return StoryResponse(**updated, chapter_count=chapter_count)

    async def delete_story(self, story_id: str, user_id: str) -> bool:
        """Safely delete a story and its associated chapters."""
        exists = await self._story_repo.exists(story_id=story_id, user_id=user_id)
        if not exists:
            raise StoryNotFoundException(story_id=story_id)

        # Cascade cleanup: remove chapters belonging to this story
        deleted_chapters = await self._chapter_repo.delete_by_story(story_id=story_id, user_id=user_id)
        logger.info(f"Cascade deleted {deleted_chapters} chapters for story ID: {story_id}")

        success = await self._story_repo.delete(story_id=story_id, user_id=user_id)
        logger.info(f"Deleted story ID: {story_id} (success={success})")
        return success


def get_story_service() -> StoryService:
    """Dependency provider for StoryService."""
    return StoryService(
        story_repo=get_story_repository(),
        chapter_repo=get_chapter_repository(),
    )
