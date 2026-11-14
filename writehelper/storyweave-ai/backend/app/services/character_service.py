"""
StoryWeave AI - Character Intelligence Service

Manages character profiles and interpersonal relationships:
- Ingests and merges character entities extracted during chapter analysis
- Enforces strict deduplication by exact canonical name and aliases
- Prevents spurious merges of distinct characters with similar names
- Tracks cross-chapter appearances (source_chapter_ids)
- Manages character profile updates and relationship graphs
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from fastapi import Depends

from app.core.logging import get_logger
from app.core.exceptions import EntityNotFoundException, StoryNotFoundException
from app.database.repositories.base import CharacterRepository, StoryRepository, AnalysisRepository
from app.database.repositories.dependencies import (
    get_character_repository,
    get_story_repository,
    get_analysis_repository,
)
from app.schemas.analysis import ChapterAnalysisResult
from app.schemas.character import (
    CharacterProfile,
    CharacterProfileResponse,
    CharacterRelationship,
    CharacterUpdate,
    CharacterListResponse,
    RelationshipListResponse,
)

logger = get_logger("service.character")


class CharacterService:
    """Service orchestrating character profile management and relationship tracking."""

    def __init__(
        self,
        character_repo: CharacterRepository,
        story_repo: StoryRepository,
        analysis_repo: Optional[AnalysisRepository] = None,
    ):
        self._character_repo = character_repo
        self._story_repo = story_repo
        self._analysis_repo = analysis_repo

    async def sync_characters_from_analysis(
        self,
        chapter_id: str,
        story_id: str,
        user_id: str,
        analysis_result: ChapterAnalysisResult,
    ) -> List[CharacterProfileResponse]:
        """
        Synchronize extracted characters and relationships from a chapter analysis.

        For each extracted character:
        1. Search for existing character in the same story by exact name or alias (case-insensitive).
        2. If found, merge new aliases, traits, abilities, motivations, goals, and append chapter_id.
        3. If not found, create a new CharacterProfile.

        For each extracted relationship:
        1. Resolve source and target characters by name or alias.
        2. If both exist and relationship does not already exist, record it.
        """
        logger.info(
            f"Syncing characters for chapter '{chapter_id}' in story '{story_id}' "
            f"({len(analysis_result.characters)} characters, {len(analysis_result.relationships)} relationships)"
        )

        synced_profiles: List[CharacterProfileResponse] = []

        # 1. Process Extracted Characters
        for char in analysis_result.characters:
            char_name = char.name.strip()
            if not char_name:
                continue

            # Check if this character already exists by primary name or alias
            existing = await self._character_repo.find_by_name_or_alias(
                story_id=story_id,
                user_id=user_id,
                name=char_name,
            )

            # If not found by primary name, check if any of its aliases match an existing character
            if not existing and char.aliases:
                for alias in char.aliases:
                    if alias.strip():
                        existing = await self._character_repo.find_by_name_or_alias(
                            story_id=story_id,
                            user_id=user_id,
                            name=alias.strip(),
                        )
                        if existing:
                            break

            now = datetime.now(timezone.utc)

            if existing:
                # Merge into existing character profile
                char_id = existing["character_id"]

                # Merge aliases (preserving uniqueness, case-preserving)
                existing_aliases = existing.get("aliases", [])
                existing_lower_aliases = {a.lower() for a in existing_aliases}
                new_aliases = list(existing_aliases)
                for a in char.aliases:
                    a_clean = a.strip()
                    if a_clean and a_clean.lower() not in existing_lower_aliases and a_clean.lower() != existing["name"].lower():
                        new_aliases.append(a_clean)
                        existing_lower_aliases.add(a_clean.lower())

                # Merge source_chapter_ids
                source_chapters = existing.get("source_chapter_ids", [])
                if chapter_id not in source_chapters:
                    source_chapters = list(source_chapters) + [chapter_id]

                # Merge traits, abilities, motivations, goals (union without duplicates)
                def _merge_unique_strings(existing_list: List[str], new_list: List[str]) -> List[str]:
                    seen = set()
                    result = []
                    for item in existing_list:
                        clean = item.strip()
                        if clean and clean.lower() not in seen:
                            seen.add(clean.lower())
                            result.append(clean)
                    for item in new_list:
                        clean = item.strip()
                        if clean and clean.lower() not in seen:
                            seen.add(clean.lower())
                            result.append(clean)
                    return result

                merged_traits = _merge_unique_strings(existing.get("personality_traits", []), char.personality_traits)
                merged_abilities = _merge_unique_strings(existing.get("abilities", []), char.abilities)
                merged_motivations = _merge_unique_strings(existing.get("motivations", []), char.motivations)
                merged_goals = _merge_unique_strings(existing.get("goals", []), char.goals)

                # Physical description: retain existing if already present, or use new if existing was empty
                desc = existing.get("physical_description") or ""
                if not desc and char.description:
                    desc = char.description.strip()

                update_payload = {
                    "aliases": new_aliases,
                    "physical_description": desc,
                    "personality_traits": merged_traits,
                    "abilities": merged_abilities,
                    "motivations": merged_motivations,
                    "goals": merged_goals,
                    "source_chapter_ids": source_chapters,
                }

                updated = await self._character_repo.update(
                    character_id=char_id,
                    user_id=user_id,
                    update_data=update_payload,
                )
                if updated:
                    synced_profiles.append(CharacterProfileResponse(**updated))
            else:
                # Create a new character profile
                char_id = f"char_{uuid.uuid4().hex[:16]}"
                clean_aliases = list(dict.fromkeys(a.strip() for a in char.aliases if a.strip() and a.strip().lower() != char_name.lower()))
                clean_traits = list(dict.fromkeys(t.strip() for t in char.personality_traits if t.strip()))
                clean_abilities = list(dict.fromkeys(ab.strip() for ab in char.abilities if ab.strip()))
                clean_motivations = list(dict.fromkeys(m.strip() for m in char.motivations if m.strip()))
                clean_goals = list(dict.fromkeys(g.strip() for g in char.goals if g.strip()))

                new_doc = {
                    "character_id": char_id,
                    "story_id": story_id,
                    "user_id": user_id,
                    "name": char_name,
                    "aliases": clean_aliases,
                    "physical_description": char.description.strip(),
                    "personality_traits": clean_traits,
                    "abilities": clean_abilities,
                    "motivations": clean_motivations,
                    "goals": clean_goals,
                    "source_chapter_ids": [chapter_id],
                    "created_at": now,
                    "updated_at": now,
                }

                saved = await self._character_repo.save_character(new_doc)
                synced_profiles.append(CharacterProfileResponse(**saved))

        # 2. Process Extracted Relationships
        for rel in analysis_result.relationships:
            src_name = rel.character_1.strip()
            tgt_name = rel.character_2.strip()
            if not src_name or not tgt_name:
                continue

            # Look up source and target characters
            src = await self._character_repo.find_by_name_or_alias(story_id=story_id, user_id=user_id, name=src_name)
            tgt = await self._character_repo.find_by_name_or_alias(story_id=story_id, user_id=user_id, name=tgt_name)

            if src and tgt:
                src_id = src["character_id"]
                tgt_id = tgt["character_id"]
                rel_type = rel.relationship_type.strip().lower()

                # Check if relationship already exists
                existing_rel = await self._character_repo.find_relationship(
                    story_id=story_id,
                    user_id=user_id,
                    source_id=src_id,
                    target_id=tgt_id,
                    rel_type=rel_type,
                )

                if not existing_rel:
                    rel_id = f"rel_{uuid.uuid4().hex[:16]}"
                    rel_doc = {
                        "relationship_id": rel_id,
                        "story_id": story_id,
                        "user_id": user_id,
                        "source_character_id": src_id,
                        "target_character_id": tgt_id,
                        "relationship_type": rel_type,
                        "description": rel.description.strip(),
                        "source_chapter_id": chapter_id,
                        "created_at": datetime.now(timezone.utc),
                    }
                    await self._character_repo.save_relationship(rel_doc)

        logger.info(f"Finished character sync for chapter '{chapter_id}': {len(synced_profiles)} profiles active.")
        return synced_profiles

    async def list_characters(self, story_id: str, user_id: str) -> CharacterListResponse:
        """List all characters belonging to a specific story."""
        story = await self._story_repo.get_by_id(story_id=story_id, user_id=user_id)
        if not story:
            raise StoryNotFoundException(story_id=story_id)

        raw_chars = await self._character_repo.list_by_story(story_id=story_id, user_id=user_id)
        profiles = [CharacterProfileResponse(**c) for c in raw_chars]
        return CharacterListResponse(
            characters=profiles,
            total=len(profiles),
            story_id=story_id,
        )

    async def get_character(self, character_id: str, user_id: str) -> CharacterProfileResponse:
        """Retrieve a specific character by ID."""
        raw_char = await self._character_repo.get_by_id(character_id=character_id, user_id=user_id)
        if not raw_char:
            raise EntityNotFoundException("Character", character_id)
        return CharacterProfileResponse(**raw_char)

    async def update_character(
        self,
        character_id: str,
        user_id: str,
        update_data: CharacterUpdate,
    ) -> CharacterProfileResponse:
        """Update fields of an existing character profile."""
        existing = await self._character_repo.get_by_id(character_id=character_id, user_id=user_id)
        if not existing:
            raise EntityNotFoundException("Character", character_id)

        update_dict = {
            k: v
            for k, v in update_data.model_dump(exclude_unset=True).items()
            if v is not None
        }

        updated = await self._character_repo.update(
            character_id=character_id,
            user_id=user_id,
            update_data=update_dict,
        )
        if not updated:
            raise EntityNotFoundException("Character", character_id)

        return CharacterProfileResponse(**updated)

    async def list_relationships(self, story_id: str, user_id: str) -> RelationshipListResponse:
        """List all character relationships belonging to a specific story."""
        story = await self._story_repo.get_by_id(story_id=story_id, user_id=user_id)
        if not story:
            raise StoryNotFoundException(story_id=story_id)

        raw_rels = await self._character_repo.list_relationships(story_id=story_id, user_id=user_id)
        rels = [CharacterRelationship(**r) for r in raw_rels]
        return RelationshipListResponse(
            relationships=rels,
            total=len(rels),
            story_id=story_id,
        )


# Singleton instance
_character_service: Optional[CharacterService] = None


def get_character_service(
    character_repo: CharacterRepository = Depends(get_character_repository),
    story_repo: StoryRepository = Depends(get_story_repository),
    analysis_repo: AnalysisRepository = Depends(get_analysis_repository),
) -> CharacterService:
    """Dependency provider for CharacterService."""
    global _character_service
    if _character_service is None:
        _character_service = CharacterService(
            character_repo=character_repo,
            story_repo=story_repo,
            analysis_repo=analysis_repo,
        )
    return _character_service
