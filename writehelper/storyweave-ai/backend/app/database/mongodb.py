"""
StoryWeave AI - MongoDB Connection Manager

Manages the MongoDB Atlas connection lifecycle with health checking,
index creation, and graceful shutdown.
"""

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from typing import Optional
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("database.mongodb")


class MongoDBManager:
    """Manages MongoDB client lifecycle and provides database access."""

    def __init__(self):
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None

    def connect(self) -> None:
        """Establish connection to MongoDB Atlas."""
        settings = get_settings()

        if not settings.is_mongodb_configured:
            logger.warning(
                "MongoDB URI not configured. Database features will be unavailable."
            )
            return

        try:
            self._client = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                maxPoolSize=50,
                minPoolSize=5,
            )

            # Force a connection to verify it works
            self._client.admin.command("ping")
            self._db = self._client[settings.database_name]

            logger.info(
                f"Connected to MongoDB Atlas — database: '{settings.database_name}'"
            )
            self._ensure_indexes()

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self._client = None
            self._db = None

    def _ensure_indexes(self) -> None:
        """Create indexes for frequently queried fields."""
        if self._db is None:
            return

        try:
            # Stories collection
            self._db.stories.create_index("user_id")
            self._db.stories.create_index([("user_id", 1), ("created_at", -1)])

            # Chapters collection
            self._db.chapters.create_index("story_id")
            self._db.chapters.create_index([("story_id", 1), ("chapter_number", 1)])
            self._db.chapters.create_index("processing_status")

            # Characters collection
            self._db.characters.create_index("story_id")
            self._db.characters.create_index(
                [("story_id", 1), ("name", 1)], unique=True
            )

            # Events collection
            self._db.events.create_index("story_id")
            self._db.events.create_index([("story_id", 1), ("chapter_id", 1)])

            # Locations collection
            self._db.locations.create_index("story_id")
            self._db.locations.create_index(
                [("story_id", 1), ("name", 1)], unique=True
            )

            # Relationships collection
            self._db.relationships.create_index("story_id")

            # Story facts collection
            self._db.story_facts.create_index("story_id")
            self._db.story_facts.create_index([("story_id", 1), ("chapter_id", 1)])

            # Analysis reports collection
            self._db.analysis_reports.create_index("chapter_id")
            self._db.analysis_reports.create_index("story_id")

            logger.info("MongoDB indexes ensured.")

        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")

    def disconnect(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            logger.info("Disconnected from MongoDB.")
        self._client = None
        self._db = None

    @property
    def db(self) -> Optional[Database]:
        """Get the database instance."""
        return self._db

    @property
    def is_connected(self) -> bool:
        """Check if MongoDB is connected and responsive."""
        if self._client is None:
            return False
        try:
            self._client.admin.command("ping")
            return True
        except Exception:
            return False

    def health_check(self) -> dict:
        """Return detailed health status for the database."""
        if self._client is None:
            settings = get_settings()
            if not settings.is_mongodb_configured:
                return {
                    "status": "not_configured",
                    "message": "MONGODB_URI environment variable is not set",
                }
            return {
                "status": "disconnected",
                "message": "MongoDB client is not connected",
            }

        try:
            self._client.admin.command("ping")
            server_info = self._client.server_info()
            return {
                "status": "healthy",
                "message": "MongoDB is connected and responsive",
                "server_version": server_info.get("version", "unknown"),
                "database": get_settings().database_name,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"MongoDB health check failed: {str(e)}",
            }


# Singleton instance
mongodb_manager = MongoDBManager()


def get_database() -> Optional[Database]:
    """Dependency: get the MongoDB database instance."""
    return mongodb_manager.db
