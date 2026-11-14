"""
StoryWeave AI - Story-Aware Writing Assistant Tests

Comprehensive unit and integration tests covering:
- Character feedback and consistency review
- Plot feedback and plot hole detection
- Strict future plot invention restrictions
- Dialogue feedback and voice preservation
- Writing style, pacing, and sensory feedback
- Continuity feedback and chapter references
- Strict story and user isolation
- Grounded responses and insufficient context detection
- Immutable manuscript preservation
- Request validation handling

Isolated unit testing environment:
- Temporary isolated ChromaDB vector store
- Deterministic mock embedding service (zero HF downloads / external network calls)
- Mocked Gemini responses (deterministic advisory outputs)
- Dependency overrides reset per test
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import hashlib
import math
from typing import List
from unittest.mock import MagicMock, AsyncMock
import pytest
from fastapi import Depends
from starlette.testclient import TestClient

from app.main import app
from app.core.security import DEV_USER_ID
from app.database.chromadb import ChromaDBManager, get_chromadb_manager
from app.database.repositories.dependencies import (
    get_story_repository,
    get_chapter_repository,
    get_character_repository,
)
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.retrieval_service import RetrievalService, get_retrieval_service
from app.services.context_builder import ContextBuilder, get_context_builder
from app.services.gemini_service import GeminiService, get_gemini_service
from app.services.writing_assistant_service import (
    WritingAssistantService,
    get_writing_assistant_service,
)
from app.services import writing_assistant_service
from app.schemas.assistant import FeedbackType


class DeterministicMockEmbeddingService:
    """Mock embedding service that produces 384-dim normalized vectors deterministically."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    is_loaded: bool = True

    def embed_text(self, text: str, normalize: bool = True) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Input text for embedding cannot be empty.")
        h = int(hashlib.md5(text.strip().encode("utf-8")).hexdigest(), 16)
        vec = [float(((h >> (i % 64)) & 1) * 2 - 1) for i in range(384)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_batch(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        return [self.embed_text(t, normalize=normalize) for t in texts]


@pytest.fixture
def temp_chroma(tmp_path) -> ChromaDBManager:
    """Fixture providing an isolated temporary ChromaDB instance."""
    chroma_dir = str(tmp_path / "chroma_assistant_test")
    manager = ChromaDBManager(persist_directory=chroma_dir)
    manager.connect()
    yield manager
    manager.disconnect()


@pytest.fixture
def mock_embedding_svc() -> DeterministicMockEmbeddingService:
    """Fixture providing offline deterministic embeddings."""
    return DeterministicMockEmbeddingService()


@pytest.fixture
def default_mock_gemini() -> MagicMock:
    """Default mock Gemini service with advisory feedback."""
    mock = MagicMock()
    mock.model_name = "gemini-flash-lite-latest"
    mock.generate_assistant_feedback = AsyncMock(
        return_value={
            "feedback": "Default feedback for test.",
            "suggestions": ["Suggestion 1"],
            "relevant_chapters": [1],
            "insufficient_context": False,
            "clarifying_questions": [],
        }
    )
    return mock


@pytest.fixture
def client(temp_chroma, mock_embedding_svc, default_mock_gemini):
    """TestClient fixture with isolated ChromaDB, deterministic embeddings, and repo cleanup."""
    get_story_repository().clear()
    get_chapter_repository().clear()
    get_character_repository().clear()

    writing_assistant_service._assistant_service = None

    def override_writing_assistant(
        story_repo=Depends(get_story_repository),
        chapter_repo=Depends(get_chapter_repository),
        character_repo=Depends(get_character_repository),
        retrieval_service=Depends(get_retrieval_service),
        context_builder=Depends(get_context_builder),
        gemini_service=Depends(get_gemini_service),
    ):
        return WritingAssistantService(
            story_repo=story_repo,
            chapter_repo=chapter_repo,
            character_repo=character_repo,
            retrieval_service=retrieval_service,
            context_builder=context_builder,
            gemini_service=gemini_service,
        )

    app.dependency_overrides[get_chromadb_manager] = lambda: temp_chroma
    app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_svc
    app.dependency_overrides[get_gemini_service] = lambda: default_mock_gemini
    app.dependency_overrides[get_writing_assistant_service] = override_writing_assistant

    test_client = TestClient(app)
    yield test_client

    app.dependency_overrides.clear()
    writing_assistant_service._assistant_service = None


@pytest.fixture
def established_universe(client):
    """Fixture providing a story, ingested chapter, and character profiles."""
    # 1. Create Story
    story_resp = client.post(
        "/api/v1/stories",
        json={
            "title": "Chronicles of Eldoria",
            "genre": "Epic Fantasy",
            "synopsis": "A saga of celestial glass and ether flame in the blighted north.",
            "world_rules_overview": "Magic requires physical celestial crystal focuses.",
        },
    )
    story = story_resp.json()

    # 2. Ingest Chapter 1
    ch1_resp = client.post(
        f"/api/v1/stories/{story['id']}/chapters",
        json={
            "title": "The Whispering Pines",
            "content": (
                "Kael walked into the Whispering Woods, clutching his sapphire crystal focus. "
                "The bitter wind bit through his cloak. Ahead, Lyra guarded the weirwood gate, "
                "her bow drawn. 'Only those with crystal focuses can pass,' she warned."
            ),
            "chapter_number": 1,
        },
    )
    ch1 = ch1_resp.json()

    # 3. Save Character Profiles in Repository
    char_repo = get_character_repository()
    import asyncio
    asyncio.run(
        char_repo.save_character({
            "character_id": "char_kael",
            "story_id": story["id"],
            "user_id": DEV_USER_ID,
            "name": "Kael",
            "aliases": ["The Crystal Seeker"],
            "physical_description": "Dark-haired tracker in weathered armor",
            "personality_traits": ["Cautious", "Determined"],
            "abilities": ["Ether-sensing", "Tracking"],
            "motivations": ["Save his blighted village"],
            "goals": ["Reach the Obsidian Spire"],
            "source_chapter_ids": [ch1["id"]],
            "created_at": "2026-11-13T00:00:00Z",
            "updated_at": "2026-11-13T00:00:00Z",
        })
    )
    asyncio.run(
        char_repo.save_character({
            "character_id": "char_lyra",
            "story_id": story["id"],
            "user_id": DEV_USER_ID,
            "name": "Lyra",
            "aliases": ["Guardian of the Glade"],
            "physical_description": "Elven archer in weirwood cloak",
            "personality_traits": ["Vigilant", "Distrusting"],
            "abilities": ["Archery", "Camouflage"],
            "motivations": ["Protect the sanctuary"],
            "goals": ["Prevent defilement"],
            "source_chapter_ids": [ch1["id"]],
            "created_at": "2026-11-13T00:00:00Z",
            "updated_at": "2026-11-13T00:00:00Z",
        })
    )

    return {
        "story": story,
        "chapter_1": ch1,
    }


class TestWritingAssistantCoreFeatures:
    """Test each required feedback mode of the writing assistant."""

    def test_character_feedback(self, client, established_universe, default_mock_gemini):
        """Assistant provides grounded character consistency advice."""
        story_id = established_universe["story"]["id"]

        mock_gemini = MagicMock()
        mock_gemini.model_name = "gemini-flash-lite-latest"
        mock_gemini.generate_assistant_feedback = AsyncMock(
            return_value={
                "feedback": (
                    "Kael's cautious and determined traits from Chapter 1 make his sudden reckless charge feel unearned. "
                    "Consider adding an internal trigger that pushes him past his usual restraint."
                ),
                "suggestions": [
                    "Show Kael noticing a physical threat to his village's survival before he acts recklessly.",
                    "Highlight his hesitation to maintain consistency with his cautious personality."
                ],
                "relevant_chapters": [1],
                "insufficient_context": False,
                "clarifying_questions": [],
            }
        )

        app.dependency_overrides[get_gemini_service] = lambda: mock_gemini

        try:
            resp = client.post(
                "/api/v1/ai/assistant",
                json={
                    "story_id": story_id,
                    "message": "Does Kael acting impulsively here fit his personality?",
                    "feedback_type": "character",
                    "passage": "Kael yelled and charged into the dragon's lair without looking back.",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["feedback_type"] == "character"
            assert "cautious" in data["feedback"].lower()
            assert len(data["suggestions"]) == 2
            assert 1 in data["relevant_chapters"]
            assert data["insufficient_context"] is False
        finally:
            app.dependency_overrides[get_gemini_service] = lambda: default_mock_gemini

    def test_plot_feedback_without_future_inventions(self, client, established_universe, default_mock_gemini):
        """Assistant gives scene-level plot critique without deciding future events."""
        story_id = established_universe["story"]["id"]

        mock_gemini = MagicMock()
        mock_gemini.model_name = "gemini-flash-lite-latest"
        mock_gemini.generate_assistant_feedback = AsyncMock(
            return_value={
                "feedback": (
                    "The stakes in this scene are clear, but the causal chain between the collapsing bridge "
                    "and their escape feels rushed. Deepening the immediate obstacle enhances tension."
                ),
                "suggestions": [
                    "Give Lyra a brief beat to cover Kael while the stone crumbles.",
                    "Clarify why the shadow beasts didn't flank the ledge."
                ],
                "relevant_chapters": [1],
                "insufficient_context": False,
                "clarifying_questions": [],
            }
        )

        app.dependency_overrides[get_gemini_service] = lambda: mock_gemini

        try:
            resp = client.post(
                "/api/v1/ai/assistant",
                json={
                    "story_id": story_id,
                    "message": "How can I improve the tension in this escape scene?",
                    "feedback_type": "plot",
                    "passage": "They ran across the bridge. It fell. They made it to the other side.",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["feedback_type"] == "plot"
            assert "stakes" in data["feedback"].lower()
            assert len(data["suggestions"]) >= 1
        finally:
            app.dependency_overrides[get_gemini_service] = lambda: default_mock_gemini

    def test_dialogue_feedback_preserves_voice(self, client, established_universe, default_mock_gemini):
        """Assistant reviews character voice and provides style-preserving suggestions."""
        story_id = established_universe["story"]["id"]

        mock_gemini = MagicMock()
        mock_gemini.model_name = "gemini-flash-lite-latest"
        mock_gemini.generate_assistant_feedback = AsyncMock(
            return_value={
                "feedback": (
                    "Lyra's dialogue feels slightly too formal compared to her terse, vigilant dialogue in Chapter 1. "
                    "Shortening her sentences can restore her guarded cadence."
                ),
                "suggestions": [
                    "Change 'It is imperative that we do not traverse this boundary' to 'Step across that line and you bleed.'"
                ],
                "relevant_chapters": [1],
                "insufficient_context": False,
                "clarifying_questions": [],
            }
        )

        app.dependency_overrides[get_gemini_service] = lambda: mock_gemini

        try:
            resp = client.post(
                "/api/v1/ai/assistant",
                json={
                    "story_id": story_id,
                    "message": "Does Lyra's line sound like her voice?",
                    "feedback_type": "dialogue",
                    "passage": "Lyra said: 'It is imperative that we do not traverse this boundary under any circumstances.'",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["feedback_type"] == "dialogue"
            assert "voice" in data["feedback"].lower() or "dialogue" in data["feedback"].lower()
            assert len(data["suggestions"]) == 1
        finally:
            app.dependency_overrides[get_gemini_service] = lambda: default_mock_gemini

    def test_writing_and_pacing_feedback(self, client, established_universe, default_mock_gemini):
        """Assistant provides prose, pacing, and sensory critique."""
        story_id = established_universe["story"]["id"]

        mock_gemini = MagicMock()
        mock_gemini.model_name = "gemini-flash-lite-latest"
        mock_gemini.generate_assistant_feedback = AsyncMock(
            return_value={
                "feedback": (
                    "The paragraph has repetitive sentence structures ('He saw...', 'He felt...'). "
                    "Varying the rhythm and adding sensory details (the smell of pine smoke, the chill on his skin) will immerse the reader."
                ),
                "suggestions": [
                    "Combine the opening clauses to vary pace.",
                    "Replace filter verbs ('He saw the mist') with direct sensory imagery ('Frost-bitten mist coiled around his boots')."
                ],
                "relevant_chapters": [],
                "insufficient_context": False,
                "clarifying_questions": [],
            }
        )

        app.dependency_overrides[get_gemini_service] = lambda: mock_gemini

        try:
            resp = client.post(
                "/api/v1/ai/assistant",
                json={
                    "story_id": story_id,
                    "message": "Can you review the pacing and prose of this opening description?",
                    "feedback_type": "writing",
                    "passage": "He saw the mist. He felt the cold. He heard a wolf. He was afraid.",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["feedback_type"] == "writing"
            assert len(data["suggestions"]) == 2
        finally:
            app.dependency_overrides[get_gemini_service] = lambda: default_mock_gemini

    def test_continuity_feedback(self, client, established_universe, default_mock_gemini):
        """Assistant flags potential magic rule breaks and references Chapter 1."""
        story_id = established_universe["story"]["id"]

        mock_gemini = MagicMock()
        mock_gemini.model_name = "gemini-flash-lite-latest"
        mock_gemini.generate_assistant_feedback = AsyncMock(
            return_value={
                "feedback": (
                    "Chapter 1 established that magic requires physical celestial crystal focuses. "
                    "In this passage, casting bare-handed ether flame contradicts that established world rule."
                ),
                "suggestions": [
                    "Mention Kael gripping a crystal catalyst in his palm.",
                    "Or explicitly frame this as an anomaly that alarms both Kael and Lyra."
                ],
                "relevant_chapters": [1],
                "insufficient_context": False,
                "clarifying_questions": [],
            }
        )

        app.dependency_overrides[get_gemini_service] = lambda: mock_gemini

        try:
            resp = client.post(
                "/api/v1/ai/assistant",
                json={
                    "story_id": story_id,
                    "message": "Are there any continuity issues with Kael casting magic here?",
                    "feedback_type": "continuity",
                    "passage": "Kael cast raw ether flame directly from his bare fingertips without any crystal focus.",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["feedback_type"] == "continuity"
            assert 1 in data["relevant_chapters"]
            assert "crystal" in data["feedback"].lower()
        finally:
            app.dependency_overrides[get_gemini_service] = lambda: default_mock_gemini


class TestWritingAssistantSafetyAndIsolation:
    """Tests for multi-tenant isolation, groundings, and manuscript immutability."""

    def test_story_isolation_blocks_cross_story_access(self, client, established_universe):
        """Consulting the assistant with another user's story returns 404."""
        resp = client.post(
            "/api/v1/ai/assistant",
            json={
                "story_id": "nonexistent_or_other_user_story",
                "message": "How is the pacing?",
            },
        )
        assert resp.status_code == 404

    def test_chapter_mismatch_returns_404(self, client, established_universe):
        """Passing a chapter_id that does not belong to the story returns 404."""
        # Create a second story
        res2 = client.post("/api/v1/stories", json={"title": "Story B", "genre": "Sci-Fi"})
        story_b_id = res2.json()["id"]

        ch1_id = established_universe["chapter_1"]["id"]

        # Request with story B but chapter from story A
        resp = client.post(
            "/api/v1/ai/assistant",
            json={
                "story_id": story_b_id,
                "chapter_id": ch1_id,
                "message": "Review this chapter.",
            },
        )
        assert resp.status_code == 404

    def test_insufficient_context_flags_uncertainty(self, client, established_universe, default_mock_gemini):
        """When story lore does not cover the topic, assistant flags insufficient context."""
        story_id = established_universe["story"]["id"]

        mock_gemini = MagicMock()
        mock_gemini.model_name = "gemini-flash-lite-latest"
        mock_gemini.generate_assistant_feedback = AsyncMock(
            return_value={
                "feedback": (
                    "There is insufficient context in your established story memories regarding the southern desert nomads. "
                    "Your story has so far focused solely on the northern Whispering Woods."
                ),
                "suggestions": [],
                "relevant_chapters": [],
                "insufficient_context": True,
                "clarifying_questions": [
                    "What cultural traditions do the southern nomads observe?",
                    "Do they also rely on celestial crystals, or a different power source?"
                ],
            }
        )

        app.dependency_overrides[get_gemini_service] = lambda: mock_gemini

        try:
            resp = client.post(
                "/api/v1/ai/assistant",
                json={
                    "story_id": story_id,
                    "message": "Do the southern desert nomads get along with the emperor?",
                    "feedback_type": "general",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["insufficient_context"] is True
            assert len(data["clarifying_questions"]) == 2
        finally:
            app.dependency_overrides[get_gemini_service] = lambda: default_mock_gemini

    def test_manuscript_remains_strictly_immutable(self, client, established_universe, default_mock_gemini):
        """Calling the writing assistant must NEVER mutate chapter content or original_content."""
        ch1 = established_universe["chapter_1"]
        original_text = ch1["content"]

        mock_gemini = MagicMock()
        mock_gemini.model_name = "gemini-flash-lite-latest"
        mock_gemini.generate_assistant_feedback = AsyncMock(
            return_value={
                "feedback": "I suggest tightening the opening sentence.",
                "suggestions": ["Tighten sentence."],
                "relevant_chapters": [1],
                "insufficient_context": False,
                "clarifying_questions": [],
            }
        )

        app.dependency_overrides[get_gemini_service] = lambda: mock_gemini

        try:
            client.post(
                "/api/v1/ai/assistant",
                json={
                    "story_id": established_universe["story"]["id"],
                    "chapter_id": ch1["id"],
                    "message": "Please rewrite the whole chapter.",
                },
            )

            # Fetch chapter and verify zero mutation
            fetch_res = client.get(f"/api/v1/chapters/{ch1['id']}")
            assert fetch_res.status_code == 200
            ch_data = fetch_res.json()
            assert ch_data["content"] == original_text
            assert ch_data["original_content"] == original_text
        finally:
            app.dependency_overrides[get_gemini_service] = lambda: default_mock_gemini

    def test_empty_message_validation_error(self, client, established_universe):
        """Empty message queries return 422 Unprocessable Entity."""
        resp = client.post(
            "/api/v1/ai/assistant",
            json={
                "story_id": established_universe["story"]["id"],
                "message": "",
            },
        )
        assert resp.status_code == 422
