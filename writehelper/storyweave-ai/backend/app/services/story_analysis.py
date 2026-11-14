"""
StoryWeave AI - Story Analysis Orchestrator Service

Orchestrates the chapter analysis workflow:
- Tracks chapter processing lifecycle (pending -> processing -> completed / failed)
- Integrates with GeminiService for deep universe knowledge extraction
- Stores structured analysis reports idempotently
- Handles retries without duplicate data
- Never overwrites the author's original manuscript
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import Depends

from app.database.repositories.base import (
    StoryRepository,
    ChapterRepository,
    AnalysisRepository,
    CharacterRepository,
)
from app.database.repositories.dependencies import (
    get_story_repository,
    get_chapter_repository,
    get_analysis_repository,
    get_character_repository,
)
from app.services.gemini_service import GeminiService, get_gemini_service
from app.services.character_service import CharacterService
from app.schemas.chapter import ProcessingStatus
from app.schemas.analysis import AnalysisReportResponse, ChapterAnalysisResult
from app.core.exceptions import (
    ChapterNotFoundException,
    StoryNotFoundException,
    EntityNotFoundException,
    StoryWeaveException,
)
from app.core.logging import get_logger

logger = get_logger("service.story_analysis")


class StoryAnalysisService:
    """Orchestrates chapter analysis and persistent memory reporting."""

    def __init__(
        self,
        story_repo: StoryRepository,
        chapter_repo: ChapterRepository,
        analysis_repo: AnalysisRepository,
        gemini_service: GeminiService,
        character_service: Optional[CharacterService] = None,
    ):
        self._story_repo = story_repo
        self._chapter_repo = chapter_repo
        self._analysis_repo = analysis_repo
        self._gemini_service = gemini_service
        self._character_service = character_service

    async def analyze_chapter(self, chapter_id: str, user_id: str) -> AnalysisReportResponse:
        """
        Execute deep universe intelligence analysis on a chapter.

        Workflow:
        1. Validate chapter and parent story exist.
        2. Set chapter status to 'processing'.
        3. Invoke Gemini for structured extraction.
        4. Save/upsert analysis report in repository.
        5. Update chapter status to 'completed'.
        6. On failure: mark status 'failed', record error message, re-raise exception.
        """
        # 1. Fetch chapter
        chapter = await self._chapter_repo.get_by_id(chapter_id=chapter_id, user_id=user_id)
        if not chapter:
            raise ChapterNotFoundException(chapter_id=chapter_id)

        story_id = chapter["story_id"]
        story = await self._story_repo.get_by_id(story_id=story_id, user_id=user_id)
        if not story:
            raise StoryNotFoundException(story_id=story_id)

        # 2. Mark chapter as PROCESSING
        await self._chapter_repo.update(
            chapter_id=chapter_id,
            user_id=user_id,
            update_data={
                "processing_status": ProcessingStatus.PROCESSING,
                "error_message": None,
            },
        )
        logger.info(f"Chapter '{chapter['title']}' (ID: {chapter_id}) status set to PROCESSING")

        # 3. Call Gemini service
        try:
            analysis_result: ChapterAnalysisResult = await self._gemini_service.analyze_chapter(
                story_context=story,
                chapter_title=chapter["title"],
                chapter_number=chapter["chapter_number"],
                content=chapter["content"],
            )

            now = datetime.now(timezone.utc)
            report_id = f"report_{uuid.uuid4().hex[:16]}"

            report_dict = {
                "id": report_id,
                "chapter_id": chapter_id,
                "story_id": story_id,
                "user_id": user_id,
                "analysis": analysis_result.model_dump(),
                "model_used": self._gemini_service.model_name,
                "created_at": now,
                "updated_at": now,
            }

            # 4. Save report (idempotent upsert via AnalysisRepository)
            saved_report = await self._analysis_repo.save_report(report_dict)

            # 4b. Automatically sync character intelligence (Phase 6)
            if self._character_service is not None:
                try:
                    await self._character_service.sync_characters_from_analysis(
                        chapter_id=chapter_id,
                        story_id=story_id,
                        user_id=user_id,
                        analysis_result=analysis_result,
                    )
                except Exception as sync_err:
                    logger.warning(
                        f"Non-fatal character sync warning for chapter {chapter_id}: {sync_err}"
                    )

            # 5. Mark chapter as COMPLETED
            await self._chapter_repo.update(
                chapter_id=chapter_id,
                user_id=user_id,
                update_data={
                    "processing_status": ProcessingStatus.COMPLETED,
                    "error_message": None,
                },
            )
            logger.info(f"Chapter ID {chapter_id} analysis COMPLETED successfully.")

            return AnalysisReportResponse(
                id=saved_report["id"],
                chapter_id=saved_report["chapter_id"],
                story_id=saved_report["story_id"],
                user_id=saved_report["user_id"],
                analysis=ChapterAnalysisResult(**saved_report["analysis"]),
                model_used=saved_report["model_used"],
                created_at=saved_report["created_at"],
                updated_at=saved_report["updated_at"],
            )

        except Exception as e:
            # 6. Mark chapter as FAILED on error
            error_msg = str(e)
            logger.error(f"Chapter ID {chapter_id} analysis FAILED: {error_msg}")

            await self._chapter_repo.update(
                chapter_id=chapter_id,
                user_id=user_id,
                update_data={
                    "processing_status": ProcessingStatus.FAILED,
                    "error_message": error_msg,
                },
            )

            if isinstance(e, StoryWeaveException):
                raise e
            raise StoryWeaveException(f"Chapter analysis pipeline failed: {error_msg}", status_code=502)

    async def get_chapter_analysis_report(self, chapter_id: str, user_id: str) -> AnalysisReportResponse:
        """Retrieve the stored analysis report for a chapter."""
        chapter = await self._chapter_repo.get_by_id(chapter_id=chapter_id, user_id=user_id)
        if not chapter:
            raise ChapterNotFoundException(chapter_id=chapter_id)

        report = await self._analysis_repo.get_by_chapter_id(chapter_id=chapter_id, user_id=user_id)
        if not report:
            raise EntityNotFoundException("AnalysisReport", chapter_id)

        return AnalysisReportResponse(
            id=report["id"],
            chapter_id=report["chapter_id"],
            story_id=report["story_id"],
            user_id=report["user_id"],
            analysis=ChapterAnalysisResult(**report["analysis"]),
            model_used=report["model_used"],
            created_at=report["created_at"],
            updated_at=report["updated_at"],
        )


def get_story_analysis_service(
    story_repo: StoryRepository = Depends(get_story_repository),
    chapter_repo: ChapterRepository = Depends(get_chapter_repository),
    analysis_repo: AnalysisRepository = Depends(get_analysis_repository),
    gemini_service: GeminiService = Depends(get_gemini_service),
    character_repo: CharacterRepository = Depends(get_character_repository),
) -> StoryAnalysisService:
    """Dependency provider for StoryAnalysisService."""
    character_service = CharacterService(
        character_repo=character_repo,
        story_repo=story_repo,
        analysis_repo=analysis_repo,
    )
    return StoryAnalysisService(
        story_repo=story_repo,
        chapter_repo=chapter_repo,
        analysis_repo=analysis_repo,
        gemini_service=gemini_service,
        character_service=character_service,
    )
