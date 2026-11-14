"""
StoryWeave AI - Custom Domain Exceptions

Provides domain-specific exceptions mapped to standardized HTTP error responses.
"""

from typing import Optional, Any, Dict


class StoryWeaveException(Exception):
    """Base exception for all domain errors."""

    def __init__(self, message: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class EntityNotFoundException(StoryWeaveException):
    """Raised when a requested resource is not found."""

    def __init__(self, entity_name: str, entity_id: str):
        super().__init__(
            message=f"{entity_name} with ID '{entity_id}' was not found.",
            status_code=404,
            details={"entity_name": entity_name, "entity_id": entity_id},
        )


class StoryNotFoundException(EntityNotFoundException):
    """Raised when a story is not found."""

    def __init__(self, story_id: str):
        super().__init__(entity_name="Story", entity_id=story_id)


class ChapterNotFoundException(EntityNotFoundException):
    """Raised when a chapter is not found."""

    def __init__(self, chapter_id: str):
        super().__init__(entity_name="Chapter", entity_id=chapter_id)


class DuplicateResourceException(StoryWeaveException):
    """Raised when attempting to create a duplicate resource."""

    def __init__(self, message: str):
        super().__init__(message=message, status_code=409)


class ValidationException(StoryWeaveException):
    """Raised for domain-level validation errors."""

    def __init__(self, message: str):
        super().__init__(message=message, status_code=400)


class GeminiNotConfiguredError(StoryWeaveException):
    """Raised when the Gemini API key is missing or not configured."""

    def __init__(self, message: str = "GEMINI_API_KEY environment variable is not configured."):
        super().__init__(message=message, status_code=503)


class GeminiAPIError(StoryWeaveException):
    """Raised when an external Gemini API call fails."""

    def __init__(self, message: str, status_code: int = 502, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=status_code, details=details)


class MalformedAnalysisResponseError(StoryWeaveException):
    """Raised when the Gemini model returns a response that cannot be parsed into schema."""

    def __init__(self, message: str, raw_output: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=502,
            details={"raw_output_snippet": raw_output[:500] if raw_output else None},
        )


class EmbeddingServiceError(StoryWeaveException):
    """Raised when embedding model loading or vector generation fails."""

    def __init__(self, message: str):
        super().__init__(message=message, status_code=500)


class MemoryServiceError(StoryWeaveException):
    """Raised when vector memory storage or search fails."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message=message, status_code=status_code)
