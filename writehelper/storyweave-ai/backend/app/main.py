"""
StoryWeave AI - Main Application Entry Point

Configures the FastAPI application with:
- CORS middleware
- Centralized exception handling
- Database lifecycle management
- API router registration
- Swagger/OpenAPI documentation
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.exceptions import StoryWeaveException
from app.database.mongodb import mongodb_manager
from app.database.chromadb import chromadb_manager

# API routers
from app.api.health import router as health_router
from app.api.stories import router as stories_router
from app.api.chapters import router as chapters_router
from app.api.characters import router as characters_router
from app.api.events import router as events_router
from app.api.memory import router as memory_router
from app.api.ai import router as ai_router
from app.api.continuity import router as continuity_router

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Connect databases
    mongodb_manager.connect()
    chromadb_manager.connect()

    logger.info("Application startup complete.")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    mongodb_manager.disconnect()
    chromadb_manager.disconnect()
    logger.info("Application shutdown complete.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "An AI-powered writing studio with Persistent Universe Intelligence. "
            "StoryWeave AI maintains deep knowledge of your fictional universe — "
            "characters, events, relationships, locations, world rules, and timeline — "
            "across your entire story using RAG-based memory retrieval."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Centralized exception handling ---

    @app.exception_handler(StoryWeaveException)
    async def storyweave_exception_handler(request: Request, exc: StoryWeaveException):
        logger.warning(f"Domain error on {request.url.path}: {exc.message}")
        content = {
            "error": exc.__class__.__name__,
            "message": exc.message,
        }
        if exc.details:
            content["details"] = exc.details
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "validation_error",
                "message": str(exc),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again later.",
            },
        )

    # --- Register routers ---

    # Health endpoints (no prefix)
    app.include_router(health_router)

    # API v1 endpoints
    app.include_router(stories_router, prefix="/api/v1")
    app.include_router(chapters_router, prefix="/api/v1")
    app.include_router(characters_router, prefix="/api/v1")
    app.include_router(events_router, prefix="/api/v1")
    app.include_router(memory_router, prefix="/api/v1")
    app.include_router(ai_router, prefix="/api/v1")
    app.include_router(continuity_router, prefix="/api/v1")

    return app


app = create_app()
