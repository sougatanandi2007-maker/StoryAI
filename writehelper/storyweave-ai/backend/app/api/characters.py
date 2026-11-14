"""
StoryWeave AI - Characters API

Character intelligence endpoints:
- List characters by story
- Get character profile by ID
- Update character profile
- List character relationships by story
"""

from fastapi import APIRouter, Depends, status
from app.core.security import get_current_user_id
from app.services.character_service import CharacterService, get_character_service
from app.schemas.character import (
    CharacterProfileResponse,
    CharacterUpdate,
    CharacterListResponse,
    RelationshipListResponse,
)

router = APIRouter(tags=["Characters"])


@router.get(
    "/stories/{story_id}/characters",
    response_model=CharacterListResponse,
    summary="List story characters",
    description="Retrieve all persistent character profiles for a story universe.",
)
async def list_characters(
    story_id: str,
    user_id: str = Depends(get_current_user_id),
    character_service: CharacterService = Depends(get_character_service),
) -> CharacterListResponse:
    return await character_service.list_characters(story_id=story_id, user_id=user_id)


@router.get(
    "/characters/{character_id}",
    response_model=CharacterProfileResponse,
    summary="Get a character profile",
    description="Retrieve detailed profile information for a character by ID.",
)
async def get_character(
    character_id: str,
    user_id: str = Depends(get_current_user_id),
    character_service: CharacterService = Depends(get_character_service),
) -> CharacterProfileResponse:
    return await character_service.get_character(character_id=character_id, user_id=user_id)


@router.put(
    "/characters/{character_id}",
    response_model=CharacterProfileResponse,
    summary="Update a character profile",
    description="Update traits, abilities, description, aliases, motivations, or goals for a character.",
)
async def update_character(
    character_id: str,
    character_in: CharacterUpdate,
    user_id: str = Depends(get_current_user_id),
    character_service: CharacterService = Depends(get_character_service),
) -> CharacterProfileResponse:
    return await character_service.update_character(
        character_id=character_id, user_id=user_id, update_data=character_in
    )


@router.get(
    "/stories/{story_id}/relationships",
    response_model=RelationshipListResponse,
    summary="List character relationships",
    description="Retrieve all observed character relationships within a story universe.",
)
async def list_relationships(
    story_id: str,
    user_id: str = Depends(get_current_user_id),
    character_service: CharacterService = Depends(get_character_service),
) -> RelationshipListResponse:
    return await character_service.list_relationships(story_id=story_id, user_id=user_id)
