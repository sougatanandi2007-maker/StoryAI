"""
StoryWeave AI - Character Schemas

Request and response validation models for Character Intelligence module.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class CharacterProfile(BaseModel):
    """Full character profile extracted from story chapters."""

    character_id: str = Field(..., description="Unique character identifier")
    story_id: str = Field(..., description="Story this character belongs to")
    user_id: str = Field(..., description="Owner user ID")
    name: str = Field(..., description="Primary character name")
    aliases: List[str] = Field(default_factory=list, description="Alternative names or titles")
    physical_description: str = Field(default="", description="Physical appearance description")
    personality_traits: List[str] = Field(default_factory=list, description="Key personality traits")
    abilities: List[str] = Field(default_factory=list, description="Skills, powers, and abilities")
    motivations: List[str] = Field(default_factory=list, description="Character motivations")
    goals: List[str] = Field(default_factory=list, description="Character goals and objectives")
    source_chapter_ids: List[str] = Field(
        default_factory=list, description="Chapters this character appeared in"
    )
    created_at: datetime = Field(..., description="Profile creation timestamp")
    updated_at: datetime = Field(..., description="Profile last updated timestamp")


class CharacterRelationship(BaseModel):
    """A directional relationship between two characters."""

    relationship_id: str = Field(..., description="Unique relationship identifier")
    story_id: str = Field(..., description="Story this relationship belongs to")
    user_id: str = Field(..., description="Owner user ID")
    source_character_id: str = Field(..., description="Source character in the relationship")
    target_character_id: str = Field(..., description="Target character in the relationship")
    relationship_type: str = Field(
        ..., description="Relationship type (ally, enemy, mentor, sibling, romantic, rival, etc.)"
    )
    description: str = Field(default="", description="Description of the relationship")
    source_chapter_id: str = Field(
        ..., description="Chapter where this relationship was observed"
    )
    created_at: datetime = Field(..., description="Relationship creation timestamp")


class CharacterProfileResponse(BaseModel):
    """Character profile representation returned by API."""

    character_id: str = Field(..., description="Unique character identifier")
    story_id: str = Field(..., description="Story this character belongs to")
    user_id: str = Field(..., description="Owner user ID")
    name: str = Field(..., description="Primary character name")
    aliases: List[str] = Field(default_factory=list, description="Alternative names or titles")
    physical_description: str = Field(default="", description="Physical appearance description")
    personality_traits: List[str] = Field(default_factory=list, description="Key personality traits")
    abilities: List[str] = Field(default_factory=list, description="Skills, powers, and abilities")
    motivations: List[str] = Field(default_factory=list, description="Character motivations")
    goals: List[str] = Field(default_factory=list, description="Character goals and objectives")
    source_chapter_ids: List[str] = Field(
        default_factory=list, description="Chapters this character appeared in"
    )
    created_at: datetime = Field(..., description="Profile creation timestamp")
    updated_at: datetime = Field(..., description="Profile last updated timestamp")

    model_config = {"from_attributes": True}


class CharacterUpdate(BaseModel):
    """Payload for updating an existing character profile. All fields optional."""

    aliases: Optional[List[str]] = Field(default=None, description="Updated aliases")
    physical_description: Optional[str] = Field(
        default=None, description="Updated physical description"
    )
    personality_traits: Optional[List[str]] = Field(
        default=None, description="Updated personality traits"
    )
    abilities: Optional[List[str]] = Field(default=None, description="Updated abilities")
    motivations: Optional[List[str]] = Field(default=None, description="Updated motivations")
    goals: Optional[List[str]] = Field(default=None, description="Updated goals")


class CharacterListResponse(BaseModel):
    """List of character profiles for a story."""

    characters: List[CharacterProfileResponse] = Field(
        ..., description="Character profiles"
    )
    total: int = Field(..., description="Total number of characters")
    story_id: str = Field(..., description="Story ID these characters belong to")


class RelationshipListResponse(BaseModel):
    """List of character relationships for a story."""

    relationships: List[CharacterRelationship] = Field(
        ..., description="Character relationships"
    )
    total: int = Field(..., description="Total number of relationships")
    story_id: str = Field(..., description="Story ID these relationships belong to")
