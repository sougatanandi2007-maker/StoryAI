"""
StoryWeave AI - Events & Timeline API (Placeholder)

Event tracking and timeline management endpoints. Full implementation in Phase 6.
"""

from fastapi import APIRouter

router = APIRouter(tags=["Events & Timeline"])


@router.get(
    "/stories/{story_id}/events",
    summary="List story events",
    status_code=501,
)
async def list_events(story_id: str):
    """List all events in a story. (Phase 6)"""
    return {"message": "Not implemented yet — coming in Phase 6"}


@router.get(
    "/stories/{story_id}/timeline",
    summary="Get story timeline",
    status_code=501,
)
async def get_timeline(story_id: str):
    """Get chronological timeline of story events. (Phase 6)"""
    return {"message": "Not implemented yet — coming in Phase 6"}
