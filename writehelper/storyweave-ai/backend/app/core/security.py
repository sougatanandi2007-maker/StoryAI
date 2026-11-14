"""
StoryWeave AI - Security Module

Placeholder for Firebase Authentication middleware.
Will be implemented after core story and memory pipelines work.
"""

from fastapi import Request, HTTPException, status
from app.core.logging import get_logger

logger = get_logger("security")

# Placeholder user ID for development (no auth required yet)
DEV_USER_ID = "dev-user-001"


async def get_current_user_id(request: Request) -> str:
    """
    Extract the authenticated user ID from the request.

    Phase 1: Returns a development user ID.
    Future: Will validate Firebase ID tokens from the Authorization header.
    """
    # TODO: Implement Firebase token verification
    # auth_header = request.headers.get("Authorization")
    # if not auth_header or not auth_header.startswith("Bearer "):
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Missing or invalid authorization header",
    #     )
    # token = auth_header.split(" ", 1)[1]
    # decoded = firebase_admin.auth.verify_id_token(token)
    # return decoded["uid"]

    logger.debug("Using development user ID (auth not yet enabled)")
    return DEV_USER_ID
