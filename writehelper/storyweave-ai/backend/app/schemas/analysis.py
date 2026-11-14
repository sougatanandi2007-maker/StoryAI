"""
StoryWeave AI - Story Analysis Schemas

Pydantic schemas for Gemini structured outputs and analysis reports.
Enforces strict typing, distinguishes explicit facts from inferred knowledge,
and mandates source text evidence.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class FactType(str, Enum):
    """Distinguishes explicitly stated facts from AI inferences."""

    EXPLICIT = "explicit"
    INFERRED = "inferred"


class ExtractedCharacter(BaseModel):
    """Character identity and traits extracted from chapter text."""

    name: str = Field(..., description="Full canonical name of the character")
    aliases: List[str] = Field(default_factory=list, description="Nicknames, titles, or alternative names")
    description: str = Field(..., description="Physical appearance and demeanor")
    personality_traits: List[str] = Field(default_factory=list, description="Core personality characteristics")
    motivations: List[str] = Field(default_factory=list, description="Observed or stated motivations")
    goals: List[str] = Field(default_factory=list, description="Active goals or intentions in this chapter")
    abilities: List[str] = Field(default_factory=list, description="Skills, magical powers, or notable competencies")
    evidence: str = Field(..., description="Direct quote or passage from the manuscript supporting this character")


class ExtractedRelationship(BaseModel):
    """Interpersonal relationship between two characters."""

    character_1: str = Field(..., description="First character name")
    character_2: str = Field(..., description="Second character name")
    relationship_type: str = Field(..., description="Type (e.g. ally, enemy, mentor, sibling, romantic, rival)")
    description: str = Field(..., description="Nature and dynamics of their relationship")
    evidence: str = Field(..., description="Manuscript evidence for this relationship")


class ExtractedEvent(BaseModel):
    """Notable narrative plot event occurring in the chapter."""

    title: str = Field(..., description="Short descriptive title of the event")
    description: str = Field(..., description="What occurred in this event")
    participants: List[str] = Field(default_factory=list, description="Characters involved in this event")
    location: Optional[str] = Field(default=None, description="Location where the event occurred")
    chronological_order: int = Field(default=1, description="Sequence order within this chapter (1, 2, 3...)")
    significance: str = Field(..., description="Narrative significance or plot impact")
    evidence: str = Field(..., description="Direct quote or passage describing the event")


class ExtractedLocation(BaseModel):
    """Setting or geographical location described in the chapter."""

    name: str = Field(..., description="Name of the place, building, realm, or location")
    description: str = Field(..., description="Sensory and spatial description of the location")
    significance: Optional[str] = Field(default=None, description="Relevance to story lore or plot")
    evidence: str = Field(..., description="Manuscript evidence describing this location")


class ExtractedFact(BaseModel):
    """Important lore, universe rule, secret, or plot fact."""

    fact: str = Field(..., description="Clear factual statement established in the text")
    category: str = Field(default="general", description="Category (lore, magic, history, tech, secret, culture)")
    fact_type: FactType = Field(
        default=FactType.EXPLICIT,
        description="Whether the fact is explicitly stated or inferred from context",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score (1.0 for explicit, lower for inferences)"
    )
    evidence: str = Field(..., description="Direct quote or manuscript excerpt supporting this fact")


class ExtractedWorldRule(BaseModel):
    """Fundamental rule governing the fictional universe."""

    rule: str = Field(..., description="The established world rule, law of physics, or magic constraint")
    category: str = Field(default="magic", description="Category (magic, physics, metaphysics, society, biology)")
    evidence: str = Field(..., description="Manuscript evidence establishing this universe rule")


class ExtractedTimelineEvent(BaseModel):
    """Chronological and temporal information extracted from the narrative."""

    relative_time: Optional[str] = Field(
        default=None,
        description="Relative time marker (e.g. 'Three days after the trial', 'At midnight', 'Ten years earlier')",
    )
    is_flashback: bool = Field(default=False, description="True if this narrative section represents a flashback")
    uncertain_dates: bool = Field(default=False, description="True if temporal placement is ambiguous or uncertain")
    time_anchor: Optional[str] = Field(default=None, description="Event or reference point this anchors to")
    evidence: str = Field(..., description="Manuscript quote supporting this timeline placement")


class ChapterSummary(BaseModel):
    """High-level summary of the chapter's narrative arc."""

    summary: str = Field(..., description="Comprehensive 2-4 paragraph narrative summary of the chapter")
    key_takeaways: List[str] = Field(default_factory=list, description="3-5 bullet points of key story beats")
    pov_character: Optional[str] = Field(default=None, description="Primary Point of View character, if identifiable")
    pacing_assessment: Optional[str] = Field(default=None, description="Observations on narrative tension and pace")


class ChapterAnalysisResult(BaseModel):
    """Complete structured intelligence extracted from a chapter by Gemini."""

    chapter_summary: ChapterSummary
    characters: List[ExtractedCharacter] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)
    events: List[ExtractedEvent] = Field(default_factory=list)
    locations: List[ExtractedLocation] = Field(default_factory=list)
    story_facts: List[ExtractedFact] = Field(default_factory=list)
    world_rules: List[ExtractedWorldRule] = Field(default_factory=list)
    timeline: List[ExtractedTimelineEvent] = Field(default_factory=list)


class AnalysisReportResponse(BaseModel):
    """API response model for a stored chapter analysis report."""

    id: str = Field(..., description="Unique analysis report ID")
    chapter_id: str = Field(..., description="Associated chapter ID")
    story_id: str = Field(..., description="Associated story universe ID")
    user_id: str = Field(..., description="Author user ID")
    analysis: ChapterAnalysisResult = Field(..., description="Full structured extraction result")
    model_used: str = Field(..., description="Gemini model that performed analysis")
    created_at: datetime = Field(..., description="Report generation timestamp")
    updated_at: datetime = Field(..., description="Report last updated timestamp")

    model_config = {"from_attributes": True}
