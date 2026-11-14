"""
StoryWeave AI - Health Check API

Provides endpoints to verify the status of the application,
database connections, and AI service availability.
"""

from fastapi import APIRouter
from datetime import datetime, timezone
from app.core.config import get_settings
from app.database.mongodb import mongodb_manager
from app.database.chromadb import chromadb_manager

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Application health check")
async def health_check():
    """Check if the StoryWeave AI application is running."""
    settings = get_settings()
    return {
        "status": "healthy",
        "application": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/database", summary="Database health check")
async def database_health():
    """Check MongoDB and ChromaDB connection status."""
    mongo_health = mongodb_manager.health_check()
    chroma_health = chromadb_manager.health_check()

    all_healthy = (
        mongo_health["status"] == "healthy"
        and chroma_health["status"] == "healthy"
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "databases": {
            "mongodb": mongo_health,
            "chromadb": chroma_health,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/ai", summary="AI services health check")
async def ai_health():
    """Check AI service configuration and availability."""
    settings = get_settings()

    gemini_status = {
        "status": "configured" if settings.is_gemini_configured else "not_configured",
        "model": settings.gemini_model,
    }

    if not settings.is_gemini_configured:
        gemini_status["message"] = "GEMINI_API_KEY environment variable is not set"
    else:
        # Quick connectivity test
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=types.HttpOptions(timeout=10000),
            )
            # Quick verification by fetching first page of models
            models_page = list(client.models.list(config={"page_size": 5}))
            gemini_status["status"] = "healthy"
            gemini_status["message"] = "Gemini API is accessible"
            gemini_status["sample_models_count"] = len(models_page)
        except Exception as e:
            gemini_status["status"] = "unhealthy"
            gemini_status["message"] = f"Gemini API check failed: {str(e)}"

    embedding_status = {
        "model": settings.embedding_model,
        "status": "available",
        "message": "Embedding model will be loaded on first use (Phase 4)",
    }

    all_healthy = gemini_status["status"] in ("healthy", "configured")

    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": {
            "gemini": gemini_status,
            "embedding": embedding_status,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
