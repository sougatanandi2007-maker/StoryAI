"""
StoryWeave AI - Story Schemas

Request and response validation models for Story management.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from app.schemas.common import PaginatedResponse


class StoryBase(BaseModel):
    """Base story fields shared across create and update schemas."""

    title: str = Field(..., min_length=1, max_length=200, description="Story title")
    synopsis: Optional[str] = Field(default=None, max_length=5000, description="Story premise / synopsis")
    genre: Optional[str] = Field(default=None, max_length=100, description="Genre (e.g. Fantasy, Sci-Fi)")
    world_rules_overview: Optional[str] = Field(
        default=None, max_length=10000, description="High-level rules of magic, technology, or universe physics"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Arbitrary author metadata (tags, target audience, tone)"
    )


class StoryCreate(StoryBase):
    """Payload for creating a new story universe."""
    pass


class StoryUpdate(BaseModel):
    """Payload for updating an existing story. All fields optional."""

    title: Optional[str] = Field(default=None, min_length=1, max_length=200, description="Updated story title")
    synopsis: Optional[str] = Field(default=None, max_length=5000, description="Updated synopsis")
    genre: Optional[str] = Field(default=None, max_length=100, description="Updated genre")
    world_rules_overview: Optional[str] = Field(
        default=None, max_length=10000, description="Updated world rules overview"
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata")


class StoryResponse(StoryBase):
    """Complete story representation returned by API."""

    id: str = Field(..., description="Unique story ID")
    user_id: str = Field(..., description="Owner user ID")
    chapter_count: int = Field(default=0, description="Total chapters in this story")
    created_at: datetime = Field(..., description="Story creation timestamp")
    updated_at: datetime = Field(..., description="Story last updated timestamp")

    model_config = {"from_attributes": True}


class PaginatedStoryResponse(PaginatedResponse[StoryResponse]):
    """Paginated list of stories."""
    pass
