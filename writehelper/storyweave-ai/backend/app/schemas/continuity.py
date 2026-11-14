"""
StoryWeave AI - Story Continuity & Consistency Schemas

Pydantic schemas for cross-chapter continuity detection, conflict evidence,
issue status tracking, and author advisory reports.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ContinuityIssueType(str, Enum):
    """Categories of narrative continuity contradictions."""

    CHARACTER_CONSISTENCY = "character_consistency"
    TIMELINE_CONSISTENCY = "timeline_consistency"
    RELATIONSHIP_CONSISTENCY = "relationship_consistency"
    WORLD_RULE_CONSISTENCY = "world_rule_consistency"
    PLOT_CONTINUITY = "plot_continuity"


class ContinuitySeverity(str, Enum):
    """Impact severity of the detected inconsistency on story cohesion."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueStatus(str, Enum):
    """Author workflow decision status for an identified continuity issue."""

    OPEN = "open"
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    RESOLVED = "resolved"


class ContinuityEvidence(BaseModel):
    """Verbatim manuscript quote or fact excerpt establishing canon."""

    chapter_id: str = Field(..., description="ID of the chapter where evidence was located")
    chapter_number: int = Field(..., description="Chapter sequence number")
    quote_or_excerpt: str = Field(..., description="Direct quote or manuscript excerpt supporting this fact")
    fact_type: str = Field(default="explicit", description="'explicit' or 'inferred'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in this evidence extraction")

    model_config = {"from_attributes": True}


class ContinuityIssue(BaseModel):
    """A single identified continuity inconsistency or contradiction between chapters."""

    issue_id: str = Field(..., description="Unique issue identifier")
    story_id: str = Field(..., description="Story universe ID")
    chapter_id: str = Field(..., description="Chapter where the contradiction was introduced")
    issue_type: ContinuityIssueType = Field(..., description="Category of continuity contradiction")
    severity: ContinuitySeverity = Field(..., description="Impact severity of the inconsistency")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score that this is a true contradiction")
    description: str = Field(..., description="Clear explanation of the continuity inconsistency")
    existing_fact: str = Field(..., description="Established canon fact from earlier chapter(s)")
    new_fact: str = Field(..., description="Conflicting fact asserted in the current chapter")
    affected_entities: List[str] = Field(default_factory=list, description="Characters, locations, or artifacts affected")
    earlier_evidence: List[ContinuityEvidence] = Field(
        default_factory=list, description="Evidence excerpts from earlier chapter(s)"
    )
    current_evidence: Optional[ContinuityEvidence] = Field(
        default=None, description="Evidence excerpt from the current chapter"
    )
    source_chapter_ids: List[str] = Field(
        default_factory=list, description="IDs of all chapters involved in this conflict"
    )
    possible_explanation: Optional[str] = Field(
        default=None,
        description="Plausible narrative explanations (e.g., character development, disguise, off-screen event)",
    )
    suggested_resolution: Optional[str] = Field(
        default=None, description="Actionable suggestion for the author to resolve or clarify"
    )
    status: IssueStatus = Field(default=IssueStatus.OPEN, description="Author decision status")
    author_notes: Optional[str] = Field(
        default=None, description="Author's notes or justification for their decision"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"from_attributes": True}


class ContinuityReport(BaseModel):
    """Comprehensive continuity audit report for a chapter against established canon."""

    report_id: str = Field(..., description="Unique report identifier")
    chapter_id: str = Field(..., description="ID of the checked chapter")
    chapter_number: int = Field(..., description="Number of the checked chapter")
    story_id: str = Field(..., description="Story universe ID")
    user_id: str = Field(..., description="Author user ID")
    total_issues: int = Field(default=0, description="Total number of issues detected")
    issues: List[ContinuityIssue] = Field(default_factory=list, description="List of identified continuity issues")
    summary: str = Field(..., description="Narrative summary of universe continuity status for this chapter")
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"from_attributes": True}


class ContinuityCheckRequest(BaseModel):
    """Request payload for running continuity checks."""

    force_recheck: bool = Field(
        default=False, description="If True, bypasses existing cached report and re-evaluates"
    )


class ContinuityCheckResponse(BaseModel):
    """API response for a continuity check execution."""

    report: ContinuityReport = Field(..., description="The continuity report")
    cached: bool = Field(default=False, description="Whether this report was returned from cache")
    message: str = Field(default="Continuity check completed successfully.")

    model_config = {"from_attributes": True}


class IssueStatusUpdateRequest(BaseModel):
    """Payload to update an issue's status and author notes."""

    status: IssueStatus = Field(
        ..., description="New status for the issue ('open', 'accepted', 'dismissed', 'resolved')"
    )
    author_notes: Optional[str] = Field(
        default=None, description="Optional notes explaining the author's decision"
    )
