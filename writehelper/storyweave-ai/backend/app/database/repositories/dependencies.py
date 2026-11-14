"""
StoryWeave AI - Repository Dependency Providers

Provides repository instances to services and route handlers via FastAPI dependency injection.
Decouples application code from concrete database engine implementations.
"""

from app.database.repositories.base import (
    StoryRepository,
    ChapterRepository,
    AnalysisRepository,
    ContinuityRepository,
    CharacterRepository,
)
from app.database.repositories.in_memory import (
    InMemoryStoryRepository,
    InMemoryChapterRepository,
    InMemoryAnalysisRepository,
    InMemoryContinuityRepository,
    InMemoryCharacterRepository,
)

# Singletons for the in-memory repositories (shared application state)
_story_repository = InMemoryStoryRepository()
_chapter_repository = InMemoryChapterRepository()
_analysis_repository = InMemoryAnalysisRepository()
_continuity_repository = InMemoryContinuityRepository()
_character_repository = InMemoryCharacterRepository()


def get_story_repository() -> StoryRepository:
    """Dependency provider for StoryRepository."""
    return _story_repository


def get_chapter_repository() -> ChapterRepository:
    """Dependency provider for ChapterRepository."""
    return _chapter_repository


def get_analysis_repository() -> AnalysisRepository:
    """Dependency provider for AnalysisRepository."""
    return _analysis_repository


def get_continuity_repository() -> ContinuityRepository:
    """Dependency provider for ContinuityRepository."""
    return _continuity_repository


def get_character_repository() -> CharacterRepository:
    """Dependency provider for CharacterRepository."""
    return _character_repository
