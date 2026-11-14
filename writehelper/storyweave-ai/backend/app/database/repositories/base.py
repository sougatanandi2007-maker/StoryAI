"""
StoryWeave AI - Abstract Repository Interfaces

Defines the contract for data persistence across Story and Chapter modules.
Decouples domain business logic from specific database engines (In-Memory, MongoDB, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any


class StoryRepository(ABC):
    """Abstract interface for Story persistence."""

    @abstractmethod
    async def create(self, story_data: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a new story document."""
        pass

    @abstractmethod
    async def get_by_id(self, story_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a story by ID, scoped to a specific user."""
        pass

    @abstractmethod
    async def list(self, user_id: str, skip: int = 0, limit: int = 20) -> Tuple[List[Dict[str, Any]], int]:
        """List stories for a user with pagination. Returns (items, total_count)."""
        pass

    @abstractmethod
    async def update(self, story_id: str, user_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update fields of an existing story."""
        pass

    @abstractmethod
    async def delete(self, story_id: str, user_id: str) -> bool:
        """Delete a story document."""
        pass

    @abstractmethod
    async def exists(self, story_id: str, user_id: str) -> bool:
        """Check whether a story exists for the given user."""
        pass


class ChapterRepository(ABC):
    """Abstract interface for Chapter persistence."""

    @abstractmethod
    async def create(self, chapter_data: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a new chapter, recording both initial and working manuscripts."""
        pass

    @abstractmethod
    async def get_by_id(self, chapter_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chapter by ID, scoped to a specific user."""
        pass

    @abstractmethod
    async def list_by_story(
        self, story_id: str, user_id: str, skip: int = 0, limit: int = 20
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List chapters belonging to a story with pagination, ordered by chapter_number."""
        pass

    @abstractmethod
    async def count_by_story(self, story_id: str, user_id: str) -> int:
        """Count total chapters in a story."""
        pass

    @abstractmethod
    async def get_by_number(self, story_id: str, chapter_number: int, user_id: str) -> Optional[Dict[str, Any]]:
        """Find a chapter by its sequence number within a story."""
        pass

    @abstractmethod
    async def update(self, chapter_id: str, user_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update chapter fields while preserving immutable original_content."""
        pass

    @abstractmethod
    async def delete(self, chapter_id: str, user_id: str) -> bool:
        """Delete a single chapter."""
        pass

    @abstractmethod
    async def delete_by_story(self, story_id: str, user_id: str) -> int:
        """Delete all chapters belonging to a story. Returns count deleted."""
        pass


class AnalysisRepository(ABC):
    """Abstract interface for Chapter Analysis Report persistence."""

    @abstractmethod
    async def save_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save or update an analysis report for a chapter (idempotent upsert)."""
        pass

    @abstractmethod
    async def get_by_chapter_id(self, chapter_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an analysis report for a specific chapter."""
        pass

    @abstractmethod
    async def delete_by_chapter_id(self, chapter_id: str, user_id: str) -> bool:
        """Delete an analysis report for a chapter."""
        pass

    @abstractmethod
    async def delete_by_story_id(self, story_id: str, user_id: str) -> int:
        """Delete all analysis reports for a story."""
        pass


class ContinuityRepository(ABC):
    """Abstract interface for Continuity Report persistence."""

    @abstractmethod
    async def save_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save or update a continuity report for a chapter (idempotent upsert)."""
        pass

    @abstractmethod
    async def get_by_id(self, report_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a continuity report by report_id."""
        pass

    @abstractmethod
    async def get_by_chapter_id(self, chapter_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the continuity report for a specific chapter."""
        pass

    @abstractmethod
    async def list_by_story(self, story_id: str, user_id: str) -> List[Dict[str, Any]]:
        """List all continuity reports for a story, ordered by chapter_number."""
        pass

    @abstractmethod
    async def update_issue_status(
        self, issue_id: str, user_id: str, status: str, author_notes: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Update the status and notes of a specific issue within a report."""
        pass

    @abstractmethod
    async def delete_by_chapter_id(self, chapter_id: str, user_id: str) -> bool:
        """Delete a continuity report for a chapter."""
        pass

    @abstractmethod
    async def delete_by_story_id(self, story_id: str, user_id: str) -> int:
        """Delete all continuity reports for a story."""
        pass


class CharacterRepository(ABC):
    """Abstract interface for Character profile and relationship persistence."""

    @abstractmethod
    async def save_character(self, char_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save a character profile (deduplicates by name within a story)."""
        pass

    @abstractmethod
    async def get_by_id(self, character_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a character profile by ID, scoped to a specific user."""
        pass

    @abstractmethod
    async def list_by_story(self, story_id: str, user_id: str) -> List[Dict[str, Any]]:
        """List all character profiles for a story."""
        pass

    @abstractmethod
    async def find_by_name(self, story_id: str, user_id: str, name: str) -> Optional[Dict[str, Any]]:
        """Find a character by exact name (case-insensitive) within a story."""
        pass

    @abstractmethod
    async def find_by_name_or_alias(self, story_id: str, user_id: str, name: str) -> Optional[Dict[str, Any]]:
        """Find a character by exact name or alias (case-insensitive) within a story."""
        pass

    @abstractmethod
    async def update(self, character_id: str, user_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update fields of an existing character profile."""
        pass

    @abstractmethod
    async def delete_by_story(self, story_id: str, user_id: str) -> int:
        """Delete all characters belonging to a story. Returns count deleted."""
        pass

    @abstractmethod
    async def save_relationship(self, rel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save a character relationship."""
        pass

    @abstractmethod
    async def list_relationships(self, story_id: str, user_id: str) -> List[Dict[str, Any]]:
        """List all character relationships for a story."""
        pass

    @abstractmethod
    async def find_relationship(
        self, story_id: str, user_id: str, source_id: str, target_id: str, rel_type: str
    ) -> Optional[Dict[str, Any]]:
        """Find a specific relationship by source, target, and type."""
        pass
