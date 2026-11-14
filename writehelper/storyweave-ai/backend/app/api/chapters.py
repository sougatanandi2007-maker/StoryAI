"""
StoryWeave AI - Chapters API

Full CRUD operations, manuscript preservation, processing lifecycle,
and deep Gemini universe intelligence analysis.
"""

from fastapi import APIRouter, Depends, Query, status
from app.core.security import get_current_user_id
from app.services.chapter_service import ChapterService, get_chapter_service
from app.services.story_analysis import StoryAnalysisService, get_story_analysis_service
from app.services.memory_service import MemoryService, get_memory_service
from app.schemas.chapter import (
    ChapterCreate,
    ChapterUpdate,
    ChapterResponse,
    PaginatedChapterResponse,
    ChapterStatusResponse,
)
from app.schemas.analysis import AnalysisReportResponse
from app.schemas.memory import IndexingReport

router = APIRouter(tags=["Chapters"])


@router.post(
    "/stories/{story_id}/chapters",
    response_model=ChapterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a new chapter",
    description=(
        "Ingest a chapter manuscript for a story. "
        "The original manuscript text is permanently preserved in 'original_content'. "
        "Initial status is set to 'pending'."
    ),
)
async def create_chapter(
    story_id: str,
    chapter_in: ChapterCreate,
    user_id: str = Depends(get_current_user_id),
    chapter_service: ChapterService = Depends(get_chapter_service),
) -> ChapterResponse:
    return await chapter_service.create_chapter(
        story_id=story_id, user_id=user_id, chapter_in=chapter_in
    )


@router.get(
    "/stories/{story_id}/chapters",
    response_model=PaginatedChapterResponse,
    summary="List chapters in a story",
    description="Retrieve a paginated list of chapters for a given story, ordered by chapter sequence number.",
)
async def list_chapters(
    story_id: str,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
    user_id: str = Depends(get_current_user_id),
    chapter_service: ChapterService = Depends(get_chapter_service),
) -> PaginatedChapterResponse:
    return await chapter_service.list_chapters(
        story_id=story_id, user_id=user_id, page=page, limit=limit
    )


@router.get(
    "/chapters/{chapter_id}",
    response_model=ChapterResponse,
    summary="Get chapter details",
    description="Retrieve a chapter by ID, including both working manuscript and original preserved content.",
)
async def get_chapter(
    chapter_id: str,
    user_id: str = Depends(get_current_user_id),
    chapter_service: ChapterService = Depends(get_chapter_service),
) -> ChapterResponse:
    return await chapter_service.get_chapter(chapter_id=chapter_id, user_id=user_id)


@router.put(
    "/chapters/{chapter_id}",
    response_model=ChapterResponse,
    summary="Update chapter draft",
    description=(
        "Update chapter title or working manuscript content. "
        "The original manuscript content ('original_content') remains immutable."
    ),
)
async def update_chapter(
    chapter_id: str,
    chapter_in: ChapterUpdate,
    user_id: str = Depends(get_current_user_id),
    chapter_service: ChapterService = Depends(get_chapter_service),
) -> ChapterResponse:
    return await chapter_service.update_chapter(
        chapter_id=chapter_id, user_id=user_id, chapter_in=chapter_in
    )


@router.delete(
    "/chapters/{chapter_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chapter",
    description="Delete a chapter and its associated metadata.",
)
async def delete_chapter(
    chapter_id: str,
    user_id: str = Depends(get_current_user_id),
    chapter_service: ChapterService = Depends(get_chapter_service),
) -> None:
    await chapter_service.delete_chapter(chapter_id=chapter_id, user_id=user_id)


@router.post(
    "/chapters/{chapter_id}/analyze",
    response_model=AnalysisReportResponse,
    summary="Analyze chapter with Gemini AI",
    description=(
        "Triggers the Gemini AI extraction pipeline to analyze chapter content. "
        "Extracts chapter summary, characters, relationships, events, locations, "
        "story facts, world rules, and timeline markers. "
        "Preserves original manuscript and updates chapter processing status."
    ),
)
async def analyze_chapter(
    chapter_id: str,
    user_id: str = Depends(get_current_user_id),
    analysis_service: StoryAnalysisService = Depends(get_story_analysis_service),
) -> AnalysisReportResponse:
    return await analysis_service.analyze_chapter(
        chapter_id=chapter_id, user_id=user_id
    )


@router.get(
    "/chapters/{chapter_id}/analysis",
    response_model=AnalysisReportResponse,
    summary="Get chapter analysis report",
    description="Retrieve the existing Gemini AI analysis report for a chapter.",
)
async def get_chapter_analysis(
    chapter_id: str,
    user_id: str = Depends(get_current_user_id),
    analysis_service: StoryAnalysisService = Depends(get_story_analysis_service),
) -> AnalysisReportResponse:
    return await analysis_service.get_chapter_analysis_report(
        chapter_id=chapter_id, user_id=user_id
    )


@router.post(
    "/chapters/{chapter_id}/retry-processing",
    response_model=AnalysisReportResponse,
    summary="Retry chapter AI analysis",
    description="Increments the chapter retry count and re-runs the Gemini analysis pipeline.",
)
async def retry_processing(
    chapter_id: str,
    user_id: str = Depends(get_current_user_id),
    chapter_service: ChapterService = Depends(get_chapter_service),
    analysis_service: StoryAnalysisService = Depends(get_story_analysis_service),
) -> AnalysisReportResponse:
    # Increment retry counter on chapter
    await chapter_service.retry_processing(chapter_id=chapter_id, user_id=user_id)
    # Execute analysis
    return await analysis_service.analyze_chapter(
        chapter_id=chapter_id, user_id=user_id
    )


@router.post(
    "/chapters/{chapter_id}/summarize",
    response_model=ChapterResponse,
    summary="Generate or retry Hugging Face chapter summarization",
    description=(
        "Sends chapter content to local DistilBART ('sshleifer/distilbart-cnn-12-6') "
        "to generate or retry a concise chapter summary. Preserves original manuscript."
    ),
)
async def summarize_chapter(
    chapter_id: str,
    user_id: str = Depends(get_current_user_id),
    chapter_service: ChapterService = Depends(get_chapter_service),
) -> ChapterResponse:
    return await chapter_service.summarize_chapter(
        chapter_id=chapter_id, user_id=user_id
    )


@router.post(
    "/chapters/{chapter_id}/index",
    response_model=IndexingReport,
    summary="Index chapter manuscript into vector memory",
    description=(
        "Splits the chapter into semantic chunks, generates vector embeddings with "
        "Sentence Transformers, and persists them into ChromaDB with source provenance metadata."
    ),
)
async def index_chapter(
    chapter_id: str,
    require_analysis: bool = Query(
        default=False,
        description="Whether to require AI analysis to be completed before indexing",
    ),
    user_id: str = Depends(get_current_user_id),
    memory_service: MemoryService = Depends(get_memory_service),
) -> IndexingReport:
    return await memory_service.index_chapter(
        chapter_id=chapter_id,
        user_id=user_id,
        require_analysis=require_analysis,
    )


@router.delete(
    "/chapters/{chapter_id}/memories",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete chapter vector memories",
    description="Deletes all indexed memory chunks for this chapter from ChromaDB.",
)
async def delete_chapter_memories(
    chapter_id: str,
    user_id: str = Depends(get_current_user_id),
    memory_service: MemoryService = Depends(get_memory_service),
) -> None:
    await memory_service.delete_chapter_memories(
        chapter_id=chapter_id, user_id=user_id
    )

