"""
StoryWeave AI - Core Configuration

Loads environment variables and provides application-wide settings
using Pydantic Settings for validation and type safety.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="StoryWeave AI", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(default="development", description="Runtime environment")
    log_level: str = Field(default="INFO", description="Logging level")

    # MongoDB
    mongodb_uri: str = Field(
        default="",
        description="MongoDB Atlas connection URI",
    )
    database_name: str = Field(
        default="storyweave",
        description="MongoDB database name",
    )

    # Google Gemini
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key",
    )
    gemini_model: str = Field(
        default="gemini-flash-lite-latest",
        description="Gemini model to use",
    )

    # ChromaDB
    chroma_persist_directory: str = Field(
        default="./data/chroma",
        description="ChromaDB persistent storage directory",
    )

    # Embedding Model
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence Transformer model for embeddings",
    )

    # Summarization Model (DistilBART)
    summarization_model: str = Field(
        default="sshleifer/distilbart-cnn-12-6",
        description="Hugging Face model for local chapter summarization",
    )
    summarization_max_chunk_words: int = Field(
        default=600,
        description="Maximum words per text chunk for summarization",
    )
    summarization_overlap_words: int = Field(
        default=50,
        description="Overlap words between chunks for long chapters",
    )

    # Firebase (optional for Phase 1)
    firebase_credentials_path: Optional[str] = Field(
        default=None,
        description="Path to Firebase service account JSON",
    )

    @property
    def is_mongodb_configured(self) -> bool:
        return bool(self.mongodb_uri)

    @property
    def is_gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def is_firebase_configured(self) -> bool:
        return bool(self.firebase_credentials_path)


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
