"""
StoryWeave AI - Chapter Model

Represents a chapter document in the database, preserving both
the initial original manuscript and the working draft.
"""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
from app.schemas.chapter import ProcessingStatus


class ChapterModel(BaseModel):
    """Internal model for a chapter document."""

    id: str
    story_id: str
    user_id: str
    title: str
    chapter_number: int
    content: str  # Working draft
    original_content: str  # Immutable initial manuscript
    word_count: int
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    retry_count: int = 0
    # Chapter Summarization (DistilBART)
    summary: Optional[str] = None
    summary_model: Optional[str] = None
    summary_status: ProcessingStatus = ProcessingStatus.PENDING
    summary_updated_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
