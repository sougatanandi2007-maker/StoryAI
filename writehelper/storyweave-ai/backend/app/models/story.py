"""
StoryWeave AI - Story Model

Represents a story universe document.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class StoryModel(BaseModel):
    """Internal model for a story universe."""

    id: str
    user_id: str
    title: str
    synopsis: Optional[str] = None
    genre: Optional[str] = None
    world_rules_overview: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
