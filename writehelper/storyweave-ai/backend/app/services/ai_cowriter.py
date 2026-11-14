"""
StoryWeave AI - AI Co-Writer & Assistant Service

Exposes WritingAssistantService and dependency provider.
"""

from app.services.writing_assistant_service import (
    WritingAssistantService,
    get_writing_assistant_service,
)

__all__ = ["WritingAssistantService", "get_writing_assistant_service"]
