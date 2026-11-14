"""
StoryWeave AI - Story-Aware Writing Assistant Schemas

Pydantic schemas for the author advisory assistant.
Provides structured validation for feedback requests (character, plot, dialogue,
writing, continuity, general) and ensures advisory responses cite relevant chapters
and maintain creative boundaries.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class FeedbackType(str, Enum):
    """Categorized feedback modes for the writing assistant."""

    CHARACTER = "character"
    PLOT = "plot"
    DIALOGUE = "dialogue"
    WRITING = "writing"
    CONTINUITY = "continuity"
    GENERAL = "general"


class AssistantChatRequest(BaseModel):
    """Payload for consulting the story-aware AI writing assistant."""

    story_id: str = Field(
        ...,
        min_length=1,
        description="Target story universe ID whose established lore and manuscripts guide feedback.",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Author's question, request for advice, or scene under review.",
    )
    feedback_type: Optional[FeedbackType] = Field(
        default=FeedbackType.GENERAL,
        description="Area of feedback focus: character, plot, dialogue, writing, continuity, or general.",
    )
    chapter_id: Optional[str] = Field(
        default=None,
        description="Optional chapter ID currently being written or evaluated.",
    )
    passage: Optional[str] = Field(
        default=None,
        max_length=10000,
        description="Optional specific draft excerpt or passage for the assistant to review.",
    )


class AssistantChatResponse(BaseModel):
    """Structured response from the writing advisor."""

    feedback: str = Field(
        ...,
        description="Detailed analytical and advisory feedback.",
    )
    story_id: str = Field(
        ...,
        description="Story universe ID.",
    )
    feedback_type: FeedbackType = Field(
        ...,
        description="Applied feedback category.",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Specific, optional creative suggestions for the author to consider.",
    )
    relevant_chapters: List[int] = Field(
        default_factory=list,
        description="Existing chapter numbers referenced in the advice or context.",
    )
    insufficient_context: bool = Field(
        default=False,
        description="True if existing story memories or context were insufficient to answer reliably.",
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="Clarifying questions for the author when context is ambiguous or unwritten.",
    )
    model_used: Optional[str] = Field(
        default=None,
        description="Gemini model identifier utilized.",
    )
