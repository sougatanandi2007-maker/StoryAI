"""
StoryWeave AI - Story Continuity API Endpoints

Provides REST endpoints for:
- Initiating automated continuity audits against established story canon.
- Retrieving chapter and story-wide continuity reports.
- Triage management: allowing authors to accept, dismiss, or resolve continuity issues.
"""

from typing import List
from fastapi import APIRouter, Depends, status

from app.core.security import get_current_user_id
from app.services.continuity_service import ContinuityService, get_continuity_service
from app.schemas.continuity import (
    ContinuityCheckRequest,
    ContinuityCheckResponse,
    ContinuityReport,
    ContinuityIssue,
    IssueStatusUpdateRequest,
)

router = APIRouter(tags=["Continuity"])


@router.post(
    "/chapters/{chapter_id}/continuity-check",
    response_model=ContinuityCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Audit chapter continuity against established canon",
    description=(
        "Analyzes the chapter against previously established story canon. "
        "Detects contradictions in characters, timeline, relationships, world rules, and plot. "
        "Returns evidence-backed issues with author suggestions. Advisory only."
    ),
)
async def check_chapter_continuity(
    chapter_id: str,
    request: ContinuityCheckRequest = ContinuityCheckRequest(),
    user_id: str = Depends(get_current_user_id),
    continuity_service: ContinuityService = Depends(get_continuity_service),
) -> ContinuityCheckResponse:
    """Run cross-chapter consistency and contradiction check."""
    return await continuity_service.check_chapter_continuity(
        chapter_id=chapter_id,
        user_id=user_id,
        force_recheck=request.force_recheck,
    )


@router.get(
    "/chapters/{chapter_id}/continuity",
    response_model=ContinuityReport,
    summary="Get chapter continuity report",
    description="Retrieves the stored continuity audit report for a chapter.",
)
async def get_chapter_continuity(
    chapter_id: str,
    user_id: str = Depends(get_current_user_id),
    continuity_service: ContinuityService = Depends(get_continuity_service),
) -> ContinuityReport:
    """Retrieve existing continuity report for a chapter."""
    return await continuity_service.get_chapter_continuity_report(
        chapter_id=chapter_id,
        user_id=user_id,
    )


@router.get(
    "/stories/{story_id}/continuity",
    response_model=List[ContinuityReport],
    summary="List all continuity reports for a story",
    description="Retrieves continuity reports for all analyzed chapters in a story, ordered by chapter sequence.",
)
async def get_story_continuity(
    story_id: str,
    user_id: str = Depends(get_current_user_id),
    continuity_service: ContinuityService = Depends(get_continuity_service),
) -> List[ContinuityReport]:
    """Retrieve all continuity reports across a story universe."""
    return await continuity_service.list_story_continuity_reports(
        story_id=story_id,
        user_id=user_id,
    )


@router.patch(
    "/continuity/issues/{issue_id}",
    response_model=ContinuityIssue,
    summary="Update continuity issue triage status",
    description=(
        "Allows authors to accept, dismiss, or resolve a detected continuity issue, "
        "and attach explanatory notes. Upholds author autonomy."
    ),
)
async def update_issue_status(
    issue_id: str,
    update_data: IssueStatusUpdateRequest,
    user_id: str = Depends(get_current_user_id),
    continuity_service: ContinuityService = Depends(get_continuity_service),
) -> ContinuityIssue:
    """Update issue status (open, accepted, dismissed, resolved) and notes."""
    return await continuity_service.update_issue_status(
        issue_id=issue_id,
        user_id=user_id,
        status=update_data.status,
        author_notes=update_data.author_notes,
    )
