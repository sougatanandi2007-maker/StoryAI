"""
StoryWeave AI - Character Intelligence Tests (Phase 6)

Comprehensive unit and integration tests covering:
- Character profile creation and retrieval
- Character profile updates via PUT
- Duplicate prevention (case-insensitive exact match)
- Non-merging of similar but distinct names ("Kael" vs "Kaela")
- Alias resolution
- Relationship extraction and retrieval
- Cross-story and cross-tenant isolation
- Automated sync during chapter analysis
- Character integration with RAG retrieval
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from starlette.testclient import TestClient

from app.main import app
from app.core.security import DEV_USER_ID
from app.database.repositories.dependencies import (
    get_story_repository,
    get_chapter_repository,
    get_character_repository,
    get_analysis_repository,
)
from app.services.gemini_service import get_gemini_service
from app.services.character_service import CharacterService, get_character_service
from app.schemas.analysis import (
    ChapterAnalysisResult,
    ChapterSummary,
    ExtractedCharacter,
    ExtractedRelationship,
)
from app.schemas.character import CharacterUpdate


@pytest.fixture
def client():
    """Test client for Character API."""
    return TestClient(app)


@pytest.fixture
def test_story(client):
    """Create a persistent test story."""
    res = client.post(
        "/api/v1/stories",
        json={"title": "Eldoria Chronicles", "genre": "Epic Fantasy", "synopsis": "A world of magic."},
    )
    assert res.status_code == 201
    return res.json()


@pytest.fixture
def test_chapter(client, test_story):
    """Create a test chapter under test_story."""
    res = client.post(
        f"/api/v1/stories/{test_story['id']}/chapters",
        json={
            "title": "The Whispering Pines",
            "content": "Kael walked into the woods clutching his sapphire pendant. Lyra watched from above.",
            "chapter_number": 1,
        },
    )
    assert res.status_code == 201
    return res.json()


class TestCharacterSyncAndDeduplication:
    """Tests for character ingestion, deduplication, and alias handling."""

    @pytest.mark.asyncio
    async def test_sync_creates_new_character(self, test_story, test_chapter):
        char_repo = get_character_repository()
        story_repo = get_story_repository()
        service = CharacterService(character_repo=char_repo, story_repo=story_repo)

        analysis = ChapterAnalysisResult(
            chapter_summary=ChapterSummary(summary="Kael enters the forest."),
            characters=[
                ExtractedCharacter(
                    name="Kael of Riverdell",
                    aliases=["The Crystal Seeker", "Kael"],
                    description="Dark hair, weathered armor",
                    personality_traits=["Determined", "Cautious"],
                    abilities=["Tracking", "Ether-sensing"],
                    motivations=["Save village"],
                    goals=["Find crystal"],
                    evidence="Kael walked into the woods",
                )
            ],
            relationships=[],
        )

        profiles = await service.sync_characters_from_analysis(
            chapter_id=test_chapter["id"],
            story_id=test_story["id"],
            user_id=DEV_USER_ID,
            analysis_result=analysis,
        )

        assert len(profiles) == 1
        p = profiles[0]
        assert p.name == "Kael of Riverdell"
        assert "The Crystal Seeker" in p.aliases
        assert "Kael" in p.aliases
        assert p.physical_description == "Dark hair, weathered armor"
        assert "Determined" in p.personality_traits
        assert "Ether-sensing" in p.abilities
        assert p.source_chapter_ids == [test_chapter["id"]]

    @pytest.mark.asyncio
    async def test_duplicate_prevention_on_resync(self, test_story, test_chapter):
        """Syncing the same chapter analysis twice must NOT duplicate the character."""
        char_repo = get_character_repository()
        story_repo = get_story_repository()
        service = CharacterService(character_repo=char_repo, story_repo=story_repo)

        analysis = ChapterAnalysisResult(
            chapter_summary=ChapterSummary(summary="Kael enters the forest."),
            characters=[
                ExtractedCharacter(
                    name="Kael",
                    aliases=["The Seeker"],
                    description="Young warrior with dark hair",
                    personality_traits=["Brave"],
                    abilities=["Swordsmanship"],
                    motivations=["Duty"],
                    goals=["Survive"],
                    evidence="Kael drew his sword",
                )
            ],
        )

        # First sync
        await service.sync_characters_from_analysis(
            chapter_id=test_chapter["id"],
            story_id=test_story["id"],
            user_id=DEV_USER_ID,
            analysis_result=analysis,
        )

        # Second sync
        await service.sync_characters_from_analysis(
            chapter_id=test_chapter["id"],
            story_id=test_story["id"],
            user_id=DEV_USER_ID,
            analysis_result=analysis,
        )

        all_chars = await char_repo.list_by_story(test_story["id"], DEV_USER_ID)
        assert len(all_chars) == 1
        assert all_chars[0]["name"] == "Kael"

    @pytest.mark.asyncio
    async def test_case_insensitive_name_matching(self, test_story, test_chapter):
        """'kael', 'KAEL', and 'Kael' must resolve to the same character."""
        char_repo = get_character_repository()
        story_repo = get_story_repository()
        service = CharacterService(character_repo=char_repo, story_repo=story_repo)

        # Ingest with lowercase
        await service.sync_characters_from_analysis(
            chapter_id=test_chapter["id"],
            story_id=test_story["id"],
            user_id=DEV_USER_ID,
            analysis_result=ChapterAnalysisResult(
                chapter_summary=ChapterSummary(summary="Intro"),
                characters=[
                    ExtractedCharacter(
                        name="kael",
                        description="A fighter",
                        evidence="kael fought",
                    )
                ],
            ),
        )

        # Ingest with uppercase in chapter 2
        await service.sync_characters_from_analysis(
            chapter_id="chap_2",
            story_id=test_story["id"],
            user_id=DEV_USER_ID,
            analysis_result=ChapterAnalysisResult(
                chapter_summary=ChapterSummary(summary="Intro"),
                characters=[
                    ExtractedCharacter(
                        name="KAEL",
                        description="A brave fighter",
                        personality_traits=["Courageous"],
                        evidence="KAEL rose again",
                    )
                ],
            ),
        )

        chars = await char_repo.list_by_story(test_story["id"], DEV_USER_ID)
        assert len(chars) == 1
        assert test_chapter["id"] in chars[0]["source_chapter_ids"]
        assert "chap_2" in chars[0]["source_chapter_ids"]
        assert "Courageous" in chars[0]["personality_traits"]

    @pytest.mark.asyncio
    async def test_similar_names_not_merged(self, test_story, test_chapter):
        """Characters with similar names (e.g. 'Kael' and 'Kaela') must NOT be merged."""
        char_repo = get_character_repository()
        story_repo = get_story_repository()
        service = CharacterService(character_repo=char_repo, story_repo=story_repo)

        analysis = ChapterAnalysisResult(
            chapter_summary=ChapterSummary(summary="Two characters meet."),
            characters=[
                ExtractedCharacter(
                    name="Kael",
                    description="Young warrior",
                    evidence="Kael spoke",
                ),
                ExtractedCharacter(
                    name="Kaela",
                    description="Elven priestess",
                    evidence="Kaela bowed",
                ),
            ],
        )

        await service.sync_characters_from_analysis(
            chapter_id=test_chapter["id"],
            story_id=test_story["id"],
            user_id=DEV_USER_ID,
            analysis_result=analysis,
        )

        chars = await char_repo.list_by_story(test_story["id"], DEV_USER_ID)
        assert len(chars) == 2
        names = {c["name"] for c in chars}
        assert names == {"Kael", "Kaela"}

    @pytest.mark.asyncio
    async def test_alias_matching_resolves_existing_character(self, test_story, test_chapter):
        """If a character is referenced by an established alias in Chapter 2, it should merge."""
        char_repo = get_character_repository()
        story_repo = get_story_repository()
        service = CharacterService(character_repo=char_repo, story_repo=story_repo)

        # Chapter 1 establishes Kael with alias "The Ghost of Eldoria"
        await service.sync_characters_from_analysis(
            chapter_id=test_chapter["id"],
            story_id=test_story["id"],
            user_id=DEV_USER_ID,
            analysis_result=ChapterAnalysisResult(
                chapter_summary=ChapterSummary(summary="Kael introduced"),
                characters=[
                    ExtractedCharacter(
                        name="Kael",
                        aliases=["The Ghost of Eldoria"],
                        description="Silent tracker",
                        evidence="They call him The Ghost of Eldoria",
                    )
                ],
            ),
        )

        # Chapter 2 refers to him as "The Ghost of Eldoria"
        await service.sync_characters_from_analysis(
            chapter_id="chap_2",
            story_id=test_story["id"],
            user_id=DEV_USER_ID,
            analysis_result=ChapterAnalysisResult(
                chapter_summary=ChapterSummary(summary="Ghost strikes"),
                characters=[
                    ExtractedCharacter(
                        name="The Ghost of Eldoria",
                        description="A cloaked phantom",
                        abilities=["Shadow-step"],
                        evidence="The Ghost of Eldoria vanished into mist",
                    )
                ],
            ),
        )

        chars = await char_repo.list_by_story(test_story["id"], DEV_USER_ID)
        assert len(chars) == 1
        assert chars[0]["name"] == "Kael"
        assert "Shadow-step" in chars[0]["abilities"]
        assert "chap_2" in chars[0]["source_chapter_ids"]


class TestCharacterAPIEndpoints:
    """REST API endpoint tests for Character Intelligence."""

    def test_list_characters_empty(self, client, test_story):
        res = client.get(f"/api/v1/stories/{test_story['id']}/characters")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 0
        assert data["characters"] == []
        assert data["story_id"] == test_story["id"]

    def test_list_characters_nonexistent_story(self, client):
        res = client.get("/api/v1/stories/nonexistent_story_123/characters")
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_get_and_update_character(self, client, test_story):
        char_repo = get_character_repository()
        char_doc = await char_repo.save_character({
            "character_id": "char_test_001",
            "story_id": test_story["id"],
            "user_id": DEV_USER_ID,
            "name": "Valen",
            "aliases": ["The Sentinel"],
            "physical_description": "Silver-haired guard",
            "personality_traits": ["Steadfast"],
            "abilities": ["Shield Mastery"],
            "motivations": ["Protect the gate"],
            "goals": ["Hold the line"],
            "source_chapter_ids": ["chap_1"],
            "created_at": "2026-11-12T00:00:00Z",
            "updated_at": "2026-11-12T00:00:00Z",
        })

        # GET by ID
        get_res = client.get(f"/api/v1/characters/{char_doc['character_id']}")
        assert get_res.status_code == 200
        data = get_res.json()
        assert data["name"] == "Valen"
        assert data["abilities"] == ["Shield Mastery"]

        # PUT update
        put_res = client.put(
            f"/api/v1/characters/{char_doc['character_id']}",
            json={
                "physical_description": "Scarred veteran with silver hair",
                "personality_traits": ["Steadfast", "World-weary"],
                "abilities": ["Shield Mastery", "Bastion Aura"],
            },
        )
        assert put_res.status_code == 200
        updated_data = put_res.json()
        assert updated_data["physical_description"] == "Scarred veteran with silver hair"
        assert updated_data["personality_traits"] == ["Steadfast", "World-weary"]
        assert updated_data["abilities"] == ["Shield Mastery", "Bastion Aura"]
        # Verify immutable fields preserved
        assert updated_data["name"] == "Valen"
        assert updated_data["character_id"] == "char_test_001"
        assert updated_data["story_id"] == test_story["id"]

    def test_get_nonexistent_character_returns_404(self, client):
        res = client.get("/api/v1/characters/nonexistent_char_999")
        assert res.status_code == 404

    def test_update_nonexistent_character_returns_404(self, client):
        res = client.put(
            "/api/v1/characters/nonexistent_char_999",
            json={"physical_description": "New description"},
        )
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_list_relationships(self, client, test_story):
        char_repo = get_character_repository()
        await char_repo.save_relationship({
            "relationship_id": "rel_001",
            "story_id": test_story["id"],
            "user_id": DEV_USER_ID,
            "source_character_id": "char_1",
            "target_character_id": "char_2",
            "relationship_type": "ally",
            "description": "Fought together against shadow beasts",
            "source_chapter_id": "chap_1",
            "created_at": "2026-11-12T00:00:00Z",
        })

        res = client.get(f"/api/v1/stories/{test_story['id']}/relationships")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 1
        assert data["relationships"][0]["relationship_type"] == "ally"
        assert data["relationships"][0]["source_character_id"] == "char_1"

    def test_list_relationships_nonexistent_story(self, client):
        res = client.get("/api/v1/stories/nonexistent_story_123/relationships")
        assert res.status_code == 404


class TestStoryIsolation:
    """Verifies strict story-level isolation for characters."""

    @pytest.mark.asyncio
    async def test_story_isolation_enforced(self, client):
        # Create Story A
        res_a = client.post("/api/v1/stories", json={"title": "Story A", "genre": "Sci-Fi"})
        story_a = res_a.json()

        # Create Story B
        res_b = client.post("/api/v1/stories", json={"title": "Story B", "genre": "Fantasy"})
        story_b = res_b.json()

        char_repo = get_character_repository()
        # Character in Story A
        char_a = await char_repo.save_character({
            "character_id": "char_story_a",
            "story_id": story_a["id"],
            "user_id": DEV_USER_ID,
            "name": "Captain Kirk",
            "created_at": "2026-11-12T00:00:00Z",
            "updated_at": "2026-11-12T00:00:00Z",
        })

        # Character in Story B
        char_b = await char_repo.save_character({
            "character_id": "char_story_b",
            "story_id": story_b["id"],
            "user_id": DEV_USER_ID,
            "name": "Gandalf",
            "created_at": "2026-11-12T00:00:00Z",
            "updated_at": "2026-11-12T00:00:00Z",
        })

        # Listing Story A must not include Story B
        list_a = client.get(f"/api/v1/stories/{story_a['id']}/characters").json()
        assert list_a["total"] == 1
        assert list_a["characters"][0]["name"] == "Captain Kirk"

        # Listing Story B must not include Story A
        list_b = client.get(f"/api/v1/stories/{story_b['id']}/characters").json()
        assert list_b["total"] == 1
        assert list_b["characters"][0]["name"] == "Gandalf"


class TestChapterAnalysisIntegration:
    """Verifies that chapter analysis automatically triggers character synchronization."""

    def test_chapter_analysis_automatically_creates_characters(self, client, test_story, test_chapter):
        mock_gemini = MagicMock()
        mock_gemini.model_name = "gemini-flash-lite-latest"
        mock_gemini.analyze_chapter = AsyncMock(
            return_value=ChapterAnalysisResult(
                chapter_summary=ChapterSummary(summary="Kael and Lyra form a wary alliance."),
                characters=[
                    ExtractedCharacter(
                        name="Kael of Riverdell",
                        aliases=["The Crystal Seeker"],
                        description="Young ether-sensing tracker with sapphire pendant",
                        personality_traits=["Cautious", "Determined"],
                        abilities=["Ether-sensing"],
                        evidence="Kael touched his glowing pendant",
                    ),
                    ExtractedCharacter(
                        name="Lyra",
                        aliases=["Guardian of the Glade"],
                        description="Elven archer in weirwood cloak",
                        personality_traits=["Vigilant"],
                        abilities=["Archery"],
                        evidence="Lyra drew her weirwood bow",
                    ),
                ],
                relationships=[
                    ExtractedRelationship(
                        character_1="Kael of Riverdell",
                        character_2="Lyra",
                        relationship_type="ally",
                        description="Wary alliance against shadow beasts",
                        evidence="Lyra lowered her bow and nodded to Kael",
                    )
                ],
            )
        )

        app.dependency_overrides[get_gemini_service] = lambda: mock_gemini

        try:
            # Trigger analysis
            analyze_res = client.post(f"/api/v1/chapters/{test_chapter['id']}/analyze")
            assert analyze_res.status_code == 200

            # Verify characters were created
            chars_res = client.get(f"/api/v1/stories/{test_story['id']}/characters")
            assert chars_res.status_code == 200
            chars_data = chars_res.json()
            assert chars_data["total"] == 2
            names = {c["name"] for c in chars_data["characters"]}
            assert names == {"Kael of Riverdell", "Lyra"}

            # Verify relationships were created
            rels_res = client.get(f"/api/v1/stories/{test_story['id']}/relationships")
            assert rels_res.status_code == 200
            rels_data = rels_res.json()
            assert rels_data["total"] == 1
            assert rels_data["relationships"][0]["relationship_type"] == "ally"

        finally:
            app.dependency_overrides.pop(get_gemini_service, None)


class TestRAGCharacterIntegration:
    """Verifies that RAG answers leverage character intelligence profiles."""

    def test_rag_incorporates_character_profile_in_chat(self, client, test_story):
        # 1. Populate a character in the story
        char_repo = get_character_repository()
        import asyncio
        asyncio.run(
            char_repo.save_character({
                "character_id": "char_kael_rag",
                "story_id": test_story["id"],
                "user_id": DEV_USER_ID,
                "name": "Kael",
                "aliases": ["The Crystal Seeker"],
                "physical_description": "Dark-haired tracker with sapphire pendant",
                "personality_traits": ["Determined", "Cautious"],
                "abilities": ["Ether-sensing", "Blade-weaving"],
                "motivations": ["Save his blighted village"],
                "goals": ["Find the Celestial Crystal"],
                "source_chapter_ids": ["chap_1"],
                "created_at": "2026-11-12T00:00:00Z",
                "updated_at": "2026-11-12T00:00:00Z",
            })
        )

        # 2. Mock Gemini to verify the prompt contains character profile info
        captured_contexts = []
        mock_gemini = MagicMock()
        mock_gemini.model_name = "gemini-flash-lite-latest"

        async def capture_rag_call(question: str, context: str):
            captured_contexts.append(context)
            return "Kael is a determined tracker whose main goal is to find the Celestial Crystal."

        mock_gemini.generate_rag_answer = AsyncMock(side_effect=capture_rag_call)
        app.dependency_overrides[get_gemini_service] = lambda: mock_gemini

        try:
            res = client.post(
                "/api/v1/ai/chat",
                json={
                    "story_id": test_story["id"],
                    "message": "What are Kael's abilities and goals?",
                },
            )
            assert res.status_code == 200
            data = res.json()
            assert "Kael" in data["answer"]

            # Verify that the context delivered to Gemini contained Kael's character profile
            assert len(captured_contexts) == 1
            prompt_context = captured_contexts[0]
            assert "RELEVANT CHARACTER PROFILES" in prompt_context
            assert "Kael" in prompt_context
            assert "Ether-sensing" in prompt_context
            assert "Find the Celestial Crystal" in prompt_context

        finally:
            app.dependency_overrides.pop(get_gemini_service, None)
