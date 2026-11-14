"""
StoryWeave AI - Chapter Management Service

Encapsulates business logic for chapter creation, retrieval, updates,
original manuscript preservation, and processing lifecycle.
"""

import math
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from app.database.repositories.base import StoryRepository, ChapterRepository
from app.database.repositories.dependencies import get_story_repository, get_chapter_repository
from app.services.summarization_service import (
    SummarizationService,
    SummarizationServiceError,
    get_summarization_service,
)
from app.schemas.chapter import (
    ChapterCreate,
    ChapterUpdate,
    ChapterResponse,
    PaginatedChapterResponse,
    ChapterStatusResponse,
    ProcessingStatus,
)
from app.core.exceptions import (
    StoryNotFoundException,
    ChapterNotFoundException,
    DuplicateResourceException,
    ValidationException,
)
from app.core.logging import get_logger

logger = get_logger("service.chapter")


class ChapterService:
    """Service handling chapter ingestion and manuscript lifecycle."""

    def __init__(
        self,
        story_repo: StoryRepository,
        chapter_repo: ChapterRepository,
        summarization_service: Optional[SummarizationService] = None,
    ):
        self._story_repo = story_repo
        self._chapter_repo = chapter_repo
        self._summarization_service = summarization_service or get_summarization_service()

    async def create_chapter(self, story_id: str, user_id: str, chapter_in: ChapterCreate) -> ChapterResponse:
        """Create and ingest a new chapter.

        Validates story existence, preserves original manuscript, triggers DistilBART
        summarization (recording status and not blocking creation if it fails), and sets status to pending.
        """
        # 1. Validate parent story exists
        story_exists = await self._story_repo.exists(story_id=story_id, user_id=user_id)
        if not story_exists:
            raise StoryNotFoundException(story_id=story_id)

        # 2. Check for duplicate chapter number in the same story
        existing_num = await self._chapter_repo.get_by_number(
            story_id=story_id,
            chapter_number=chapter_in.chapter_number,
            user_id=user_id,
        )
        if existing_num:
            raise DuplicateResourceException(
                f"Chapter {chapter_in.chapter_number} already exists in this story. "
                f"Choose a unique chapter number or update the existing chapter."
            )

        now = datetime.now(timezone.utc)
        chapter_id = f"chap_{uuid.uuid4().hex[:16]}"
        content_text = chapter_in.content.strip()
        word_count = len(content_text.split())

        # 3. Summarization with DistilBART (non-blocking if failure occurs)
        summary_text: Optional[str] = None
        summary_model_name: Optional[str] = self._summarization_service.model_name
        summary_status = ProcessingStatus.PENDING

        try:
            logger.info(f"Generating DistilBART summary for new chapter {chapter_in.chapter_number}...")
            summary_text = self._summarization_service.summarize_chapter(content_text)
            summary_status = ProcessingStatus.COMPLETED
            logger.info(f"DistilBART summary generated successfully for chapter {chapter_in.chapter_number}")
        except Exception as e:
            summary_status = ProcessingStatus.FAILED
            logger.error(
                f"DistilBART summarization failed for chapter {chapter_in.chapter_number}: {e}. "
                f"Original manuscript preserved; chapter creation continues."
            )

        # 4. Store document with immutable original_content snapshot
        chapter_dict = {
            "id": chapter_id,
            "story_id": story_id,
            "user_id": user_id,
            "title": chapter_in.title.strip(),
            "chapter_number": chapter_in.chapter_number,
            "content": content_text,
            "original_content": content_text,  # Permanently preserved
            "word_count": word_count,
            "processing_status": ProcessingStatus.PENDING,
            "error_message": None,
            "retry_count": 0,
            "summary": summary_text,
            "summary_model": summary_model_name,
            "summary_status": summary_status,
            "summary_updated_at": now,
            "created_at": now,
            "updated_at": now,
        }

        created = await self._chapter_repo.create(chapter_dict)
        logger.info(
            f"Created chapter {chapter_in.chapter_number} ('{chapter_in.title}') in story {story_id} "
            f"with {word_count} words (Status: PENDING, SummaryStatus: {summary_status})"
        )
        return ChapterResponse(**created)

    async def get_chapter(self, chapter_id: str, user_id: str) -> ChapterResponse:
        """Retrieve a specific chapter by ID."""
        chapter = await self._chapter_repo.get_by_id(chapter_id=chapter_id, user_id=user_id)
        if not chapter:
            raise ChapterNotFoundException(chapter_id=chapter_id)
        return ChapterResponse(**chapter)

    async def list_chapters(
        self, story_id: str, user_id: str, page: int = 1, limit: int = 20
    ) -> PaginatedChapterResponse:
        """List all chapters in a story ordered by sequence number."""
        story_exists = await self._story_repo.exists(story_id=story_id, user_id=user_id)
        if not story_exists:
            raise StoryNotFoundException(story_id=story_id)

        skip = (page - 1) * limit
        items_raw, total = await self._chapter_repo.list_by_story(
            story_id=story_id, user_id=user_id, skip=skip, limit=limit
        )

        items = [ChapterResponse(**ch) for ch in items_raw]
        total_pages = math.ceil(total / limit) if total > 0 else 1

        return PaginatedChapterResponse(
            items=items,
            total=total,
            page=page,
            limit=limit,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )

    async def update_chapter(self, chapter_id: str, user_id: str, chapter_in: ChapterUpdate) -> ChapterResponse:
        """Update chapter details.

        Preserves original_content while updating the working draft in 'content'.
        """
        existing = await self._chapter_repo.get_by_id(chapter_id=chapter_id, user_id=user_id)
        if not existing:
            raise ChapterNotFoundException(chapter_id=chapter_id)

        update_dict = chapter_in.model_dump(exclude_unset=True)

        # Check chapter_number uniqueness if changing
        if "chapter_number" in update_dict and update_dict["chapter_number"] != existing["chapter_number"]:
            conflict = await self._chapter_repo.get_by_number(
                story_id=existing["story_id"],
                chapter_number=update_dict["chapter_number"],
                user_id=user_id,
            )
            if conflict and conflict["id"] != chapter_id:
                raise DuplicateResourceException(
                    f"Chapter {update_dict['chapter_number']} already exists in this story."
                )

        if "title" in update_dict and update_dict["title"]:
            update_dict["title"] = update_dict["title"].strip()

        if "content" in update_dict and update_dict["content"]:
            update_dict["content"] = update_dict["content"].strip()

        updated = await self._chapter_repo.update(
            chapter_id=chapter_id, user_id=user_id, update_data=update_dict
        )
        logger.info(f"Updated chapter ID: {chapter_id} (original manuscript remains intact)")
        return ChapterResponse(**updated)

    async def delete_chapter(self, chapter_id: str, user_id: str) -> bool:
        """Delete a chapter."""
        existing = await self._chapter_repo.get_by_id(chapter_id=chapter_id, user_id=user_id)
        if not existing:
            raise ChapterNotFoundException(chapter_id=chapter_id)

        success = await self._chapter_repo.delete(chapter_id=chapter_id, user_id=user_id)
        logger.info(f"Deleted chapter ID: {chapter_id} (success={success})")
        return success

    async def trigger_analysis(self, chapter_id: str, user_id: str) -> ChapterStatusResponse:
        """Mark chapter processing status as pending for AI analysis.

        In Phase 2, validates chapter exists and sets status to pending.
        Phase 3 will connect this to Gemini chapter analysis pipeline.
        """
        existing = await self._chapter_repo.get_by_id(chapter_id=chapter_id, user_id=user_id)
        if not existing:
            raise ChapterNotFoundException(chapter_id=chapter_id)

        updated = await self._chapter_repo.update(
            chapter_id=chapter_id,
            user_id=user_id,
            update_data={"processing_status": ProcessingStatus.PENDING},
        )

        return ChapterStatusResponse(
            chapter_id=chapter_id,
            story_id=existing["story_id"],
            processing_status=ProcessingStatus.PENDING,
            retry_count=existing.get("retry_count", 0),
            error_message=None,
            message="Chapter queued for AI memory extraction and universe analysis.",
        )

    async def retry_processing(self, chapter_id: str, user_id: str) -> ChapterStatusResponse:
        """Retry failed chapter processing."""
        existing = await self._chapter_repo.get_by_id(chapter_id=chapter_id, user_id=user_id)
        if not existing:
            raise ChapterNotFoundException(chapter_id=chapter_id)

        new_retry_count = existing.get("retry_count", 0) + 1
        updated = await self._chapter_repo.update(
            chapter_id=chapter_id,
            user_id=user_id,
            update_data={
                "processing_status": ProcessingStatus.PENDING,
                "retry_count": new_retry_count,
                "error_message": None,
            },
        )

        return ChapterStatusResponse(
            chapter_id=chapter_id,
            story_id=existing["story_id"],
            processing_status=ProcessingStatus.PENDING,
            retry_count=new_retry_count,
            error_message=None,
            message=f"Chapter processing retry #{new_retry_count} queued successfully.",
        )

    async def summarize_chapter(self, chapter_id: str, user_id: str) -> ChapterResponse:
        """Explicitly run or retry DistilBART summarization for a chapter.

        Preserves the original manuscript and updates summary fields.
        """
        existing = await self._chapter_repo.get_by_id(chapter_id=chapter_id, user_id=user_id)
        if not existing:
            raise ChapterNotFoundException(chapter_id=chapter_id)

        content = existing.get("content", "")
        if not content or not content.strip():
            raise ValidationException("Chapter content is empty; cannot generate summary.")

        now = datetime.now(timezone.utc)
        try:
            logger.info(f"Running summarization retry/generation for chapter {chapter_id}...")
            summary_text = self._summarization_service.summarize_chapter(content)
            updated = await self._chapter_repo.update(
                chapter_id=chapter_id,
                user_id=user_id,
                update_data={
                    "summary": summary_text,
                    "summary_model": self._summarization_service.model_name,
                    "summary_status": ProcessingStatus.COMPLETED,
                    "summary_updated_at": now,
                },
            )
            logger.info(f"Chapter {chapter_id} summarization succeeded.")
            return ChapterResponse(**updated)
        except Exception as e:
            logger.error(f"Chapter {chapter_id} summarization failed: {e}")
            updated = await self._chapter_repo.update(
                chapter_id=chapter_id,
                user_id=user_id,
                update_data={
                    "summary_status": ProcessingStatus.FAILED,
                    "summary_updated_at": now,
                },
            )
            raise SummarizationServiceError(f"Summarization failed for chapter {chapter_id}: {str(e)}")


def get_chapter_service() -> ChapterService:
    """Dependency provider for ChapterService."""
    return ChapterService(
        story_repo=get_story_repository(),
        chapter_repo=get_chapter_repository(),
        summarization_service=get_summarization_service(),
    )
