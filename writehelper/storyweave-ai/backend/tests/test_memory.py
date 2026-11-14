"""
Tests for StoryWeave AI Phase 4 - Part A: Story Memory Engine.

Covers:
- Embedding service (SentenceTransformers all-MiniLM-L6-v2, 384 dimensions, batching, validation)
- Chapter chunking service (sentence boundaries, overlap, provenance metadata, deterministic IDs)
- ChromaDB vector store (upsert, idempotency, cosine similarity scoring, user/story isolation, deletion)
- Memory service orchestration (index_chapter, search_memories, get_memories, get_memory_status)
- REST API endpoints (/api/v1/chapters/{id}/index, /api/v1/memory/search, /api/v1/stories/{id}/memories, status)
"""

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.chunking_service import ChunkingService, get_chunking_service
from app.database.chromadb import ChromaDBManager, get_chromadb_manager
from app.services.memory_service import MemoryService, get_memory_service
from app.schemas.memory import MemoryType
from app.core.exceptions import ChapterNotFoundException, StoryNotFoundException, ValidationException


# --- Fixtures ---

@pytest.fixture
def embedding_svc() -> EmbeddingService:
    """Fixture providing singleton EmbeddingService."""
    return get_embedding_service()


@pytest.fixture
def chunking_svc() -> ChunkingService:
    """Fixture providing ChunkingService."""
    return ChunkingService(target_chunk_size=400, chunk_overlap=100)


@pytest.fixture
def temp_chroma(tmp_path) -> ChromaDBManager:
    """Fixture providing an isolated ChromaDB instance using pytest tmp_path."""
    chroma_dir = str(tmp_path / "chroma_test")
    manager = ChromaDBManager(persist_directory=chroma_dir)
    manager.connect()
    yield manager
    manager.disconnect()


@pytest.fixture
def client_with_chroma(tmp_path):
    """FastAPI TestClient with isolated ChromaDB storage."""
    chroma_dir = str(tmp_path / "chroma_api_test")
    manager = ChromaDBManager(persist_directory=chroma_dir)
    manager.connect()

    app.dependency_overrides[get_chromadb_manager] = lambda: manager
    client = TestClient(app)
    yield client, manager

    app.dependency_overrides.clear()
    manager.disconnect()


# --- 1. Embedding Service Tests ---

class TestEmbeddingService:
    def test_embed_single_text(self, embedding_svc: EmbeddingService):
        text = "The ancient celestial crystal resonated with raw ether magic."
        vector = embedding_svc.embed_text(text)

        assert isinstance(vector, list)
        assert len(vector) == 384
        arr = np.array(vector)
        assert np.all(np.isfinite(arr))
        # Normalized L2 norm should be ~ 1.0
        assert math.isclose(np.linalg.norm(arr), 1.0, rel_tol=1e-3)

    def test_embed_batch(self, embedding_svc: EmbeddingService):
        texts = [
            "Kael gripped his crystal dagger in the frozen grove.",
            "Lyra loosed a silver-fletched arrow from the canopy.",
            "The Blood Eclipse approached, turning the clouds crimson.",
        ]
        vectors = embedding_svc.embed_batch(texts)

        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == 384
            assert np.all(np.isfinite(v))
            assert math.isclose(np.linalg.norm(v), 1.0, rel_tol=1e-3)

    def test_embed_empty_text_raises_validation(self, embedding_svc: EmbeddingService):
        with pytest.raises(ValueError):
            embedding_svc.embed_text("")

        with pytest.raises(ValueError):
            embedding_svc.embed_text("   \n\t  ")

    def test_embed_batch_with_invalid_item_raises(self, embedding_svc: EmbeddingService):
        with pytest.raises(ValueError):
            embedding_svc.embed_batch([])

        with pytest.raises(ValueError):
            embedding_svc.embed_batch(["Valid text", "", "Another valid text"])

    def test_model_is_loaded_and_dimension_cached(self, embedding_svc: EmbeddingService):
        assert embedding_svc.is_loaded
        assert embedding_svc.dimension == 384
        assert embedding_svc.model_name == "sentence-transformers/all-MiniLM-L6-v2"


# --- 2. Chunking Service Tests ---

class TestChunkingService:
    def test_chunking_preserves_sentences_and_generates_ids(self, chunking_svc: ChunkingService):
        manuscript = (
            "The wind howled through the twisted pines of Whispering Woods. "
            "Kael clutching his crystal dagger, stepped carefully over the frozen roots. "
            "From the canopy above, an arrow struck the dirt inches from Kael's boot. "
            "'Turn back, mortal,' Lyra warned, her silver-threaded cloak shimmering like starlight. "
            "Three shadow beasts sprang from the darkness, their crimson eyes flashing. "
            "'If you wish to live through the eclipse,' Lyra muttered, 'we fight together.' "
            "When the moon turns red, the ancient wards crumble like ash. "
            "No spell could be woven without channeling the energy through refined celestial glass."
        )

        chunks = chunking_svc.chunk_chapter(
            chapter_id="chap_123",
            story_id="story_456",
            user_id="default_user",
            chapter_number=1,
            content=manuscript,
        )

        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks):
            # Check deterministic IDs
            assert chunk.id == f"chap_123_chunk_{i}"
            assert chunk.text.strip() != ""
            # Check metadata fields
            assert chunk.metadata["user_id"] == "default_user"
            assert chunk.metadata["story_id"] == "story_456"
            assert chunk.metadata["chapter_id"] == "chap_123"
            assert chunk.metadata["chapter_number"] == 1
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["memory_type"] == MemoryType.MANUSCRIPT_CHUNK.value
            assert chunk.metadata["word_count"] > 0
            assert "source_text" in chunk.metadata

    def test_chunking_empty_content(self, chunking_svc: ChunkingService):
        chunks = chunking_svc.chunk_chapter(
            chapter_id="chap_empty",
            story_id="story_empty",
            user_id="default_user",
            chapter_number=1,
            content="",
        )
        assert chunks == []

    def test_chunking_no_manuscript_loss(self, chunking_svc: ChunkingService):
        """Verify key sentences from original manuscript appear in chunks."""
        manuscript = (
            "The first sentence sets the stage. The second sentence introduces conflict. "
            "The third sentence reveals the ancient secret of Eldoria. "
            "Finally, the fourth sentence concludes this tense chapter."
        )
        chunks = chunking_svc.chunk_chapter(
            chapter_id="chap_loss_test",
            story_id="story_loss_test",
            user_id="default_user",
            chapter_number=1,
            content=manuscript,
        )
        all_chunk_text = " ".join(c.text for c in chunks)
        assert "first sentence sets the stage" in all_chunk_text
        assert "ancient secret of Eldoria" in all_chunk_text
        assert "concludes this tense chapter" in all_chunk_text


# --- 3. ChromaDB Manager Tests ---

class TestChromaDBManager:
    def test_upsert_and_retrieve_memories(self, temp_chroma: ChromaDBManager, embedding_svc: EmbeddingService):
        doc = "Lyra the elven archer guarded the sanctum of Eldoria."
        vec = embedding_svc.embed_text(doc)

        count = temp_chroma.upsert_chunks(
            ids=["chunk_test_1"],
            documents=[doc],
            embeddings=[vec],
            metadatas=[{
                "user_id": "user_a",
                "story_id": "story_1",
                "chapter_id": "chap_1",
                "chapter_number": 1,
                "chunk_index": 0,
                "memory_type": "manuscript_chunk",
            }],
        )
        assert count == 1

        items, total = temp_chroma.get_memories_for_story(story_id="story_1", user_id="user_a")
        assert total == 1
        assert items[0]["id"] == "chunk_test_1"
        assert items[0]["text"] == doc

    def test_reindexing_idempotency(self, temp_chroma: ChromaDBManager, embedding_svc: EmbeddingService):
        """Upserting the same chunk ID twice updates in place without duplicating."""
        doc1 = "Initial manuscript version."
        doc2 = "Updated manuscript version with edits."
        vec1 = embedding_svc.embed_text(doc1)
        vec2 = embedding_svc.embed_text(doc2)

        temp_chroma.upsert_chunks(
            ids=["chap_1_chunk_0"],
            documents=[doc1],
            embeddings=[vec1],
            metadatas=[{"user_id": "u1", "story_id": "s1", "chapter_id": "c1"}],
        )
        # Upsert again with same ID
        temp_chroma.upsert_chunks(
            ids=["chap_1_chunk_0"],
            documents=[doc2],
            embeddings=[vec2],
            metadatas=[{"user_id": "u1", "story_id": "s1", "chapter_id": "c1"}],
        )

        items, total = temp_chroma.get_memories_for_story(story_id="s1", user_id="u1")
        assert total == 1
        assert items[0]["text"] == doc2

    def test_semantic_search_with_similarity_score(
        self, temp_chroma: ChromaDBManager, embedding_svc: EmbeddingService
    ):
        docs = [
            ("The guardian archer aimed her silver bow at the intruders.", "bow_chunk"),
            ("The alchemist boiled nightshade root over the blue fire.", "alchemy_chunk"),
            ("A fleet of starships entered hyperspace above the desert planet.", "scifi_chunk"),
        ]
        ids = [d[1] for d in docs]
        texts = [d[0] for d in docs]
        embeddings = embedding_svc.embed_batch(texts)
        metadatas = [
            {"user_id": "u1", "story_id": "fantasy_story", "chapter_id": "c1", "memory_type": "manuscript_chunk"}
            for _ in docs
        ]

        temp_chroma.upsert_chunks(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

        query = "Who was shooting arrows in the forest?"
        q_vec = embedding_svc.embed_text(query)

        results = temp_chroma.search(
            query_embedding=q_vec,
            user_id="u1",
            story_id="fantasy_story",
            top_k=3,
        )

        assert len(results) >= 1
        top_match = results[0]
        # Most relevant match must be the guardian archer chunk
        assert top_match["id"] == "bow_chunk"
        assert 0.0 <= top_match["similarity_score"] <= 1.0
        assert top_match["similarity_score"] > 0.4

    def test_story_isolation(self, temp_chroma: ChromaDBManager, embedding_svc: EmbeddingService):
        """Memories belonging to Story A must never appear in Story B search."""
        doc_a = "The sacred dragon egg hatched in Story Alpha."
        doc_b = "A murder occurred in the cyber tavern of Story Beta."

        temp_chroma.upsert_chunks(
            ids=["chunk_alpha"],
            documents=[doc_a],
            embeddings=[embedding_svc.embed_text(doc_a)],
            metadatas=[{"user_id": "u1", "story_id": "story_alpha", "chapter_id": "ca"}],
        )
        temp_chroma.upsert_chunks(
            ids=["chunk_beta"],
            documents=[doc_b],
            embeddings=[embedding_svc.embed_text(doc_b)],
            metadatas=[{"user_id": "u1", "story_id": "story_beta", "chapter_id": "cb"}],
        )

        # Search story_beta for dragon egg
        results = temp_chroma.search(
            query_embedding=embedding_svc.embed_text("dragon egg"),
            user_id="u1",
            story_id="story_beta",
            top_k=5,
        )
        # Should NOT return chunk_alpha
        result_ids = [r["id"] for r in results]
        assert "chunk_alpha" not in result_ids

    def test_user_isolation(self, temp_chroma: ChromaDBManager, embedding_svc: EmbeddingService):
        """Memories belonging to User A must never appear in User B search."""
        doc = "Confidential plot twist belonging to user Alice."
        temp_chroma.upsert_chunks(
            ids=["alice_secret"],
            documents=[doc],
            embeddings=[embedding_svc.embed_text(doc)],
            metadatas=[{"user_id": "user_alice", "story_id": "shared_story_id", "chapter_id": "c1"}],
        )

        results = temp_chroma.search(
            query_embedding=embedding_svc.embed_text("plot twist"),
            user_id="user_bob",  # Different user
            story_id="shared_story_id",
            top_k=5,
        )
        assert len(results) == 0

    def test_delete_by_chapter(self, temp_chroma: ChromaDBManager, embedding_svc: EmbeddingService):
        temp_chroma.upsert_chunks(
            ids=["c1_chunk_0", "c2_chunk_0"],
            documents=["Chapter 1 content", "Chapter 2 content"],
            embeddings=embedding_svc.embed_batch(["Chapter 1 content", "Chapter 2 content"]),
            metadatas=[
                {"user_id": "u1", "story_id": "s1", "chapter_id": "c1"},
                {"user_id": "u1", "story_id": "s1", "chapter_id": "c2"},
            ],
        )

        temp_chroma.delete_by_chapter(chapter_id="c1", user_id="u1")
        items, total = temp_chroma.get_memories_for_story(story_id="s1", user_id="u1")
        assert total == 1
        assert items[0]["id"] == "c2_chunk_0"


# --- 4. API Endpoints Integration Tests ---

class TestMemoryAPIEndpoints:
    def test_index_chapter_and_search_workflow(self, client_with_chroma):
        client, chroma = client_with_chroma

        # 1. Create Story
        story_res = client.post(
            "/api/v1/stories",
            json={
                "title": "The Whispering Grove",
                "synopsis": "An archer and an ether-mage seek celestial crystals.",
                "genre": "Fantasy",
            },
        )
        assert story_res.status_code == 201
        story_id = story_res.json()["id"]

        # 2. Create Chapter
        manuscript = (
            "The frost clinging to the pines of Whispering Woods crackled under Kael's boots. "
            "He drew the sapphire-hued crystal dagger, feeling its subtle pulse against his palm. "
            "High in the canopy, Lyra pulled back her weirwood bow. Her silver-threaded cloak "
            "blended seamlessly with the mist. 'State your purpose, trespasser,' she whispered. "
            "Before Kael could reply, two crimson-eyed shadow beasts leaped from the thicket. "
            "Lyra's arrow struck the first beast mid-air with a blinding flash of celestial light."
        )
        chap_res = client.post(
            f"/api/v1/stories/{story_id}/chapters",
            json={
                "title": "Encounter in the Pines",
                "content": manuscript,
                "chapter_number": 1,
            },
        )
        assert chap_res.status_code == 201
        chapter_id = chap_res.json()["id"]

        # 3. Index Chapter
        index_res = client.post(f"/api/v1/chapters/{chapter_id}/index")
        assert index_res.status_code == 200
        index_data = index_res.json()
        assert index_data["chapter_id"] == chapter_id
        assert index_data["story_id"] == story_id
        assert index_data["chunks_indexed"] >= 1
        assert index_data["status"] == "completed"

        # 4. Check Story Memory Status
        status_res = client.get(f"/api/v1/stories/{story_id}/memory-status")
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["story_id"] == story_id
        assert status_data["is_indexed"] is True
        assert status_data["total_chunks"] >= 1
        assert chapter_id in status_data["indexed_chapter_ids"]

        # 5. List Story Memories
        list_res = client.get(f"/api/v1/stories/{story_id}/memories")
        assert list_res.status_code == 200
        list_data = list_res.json()
        assert list_data["total"] >= 1
        assert len(list_data["items"]) >= 1
        assert list_data["items"][0]["metadata"]["chapter_id"] == chapter_id

        # 6. Semantic Search
        search_res = client.post(
            "/api/v1/memory/search",
            json={
                "story_id": story_id,
                "query": "What weapon did Lyra use against the shadow beast?",
                "top_k": 3,
            },
        )
        assert search_res.status_code == 200
        search_data = search_res.json()
        assert search_data["story_id"] == story_id
        assert search_data["total_found"] >= 1
        first_match = search_data["results"][0]
        assert "Lyra" in first_match["text"]
        assert first_match["similarity_score"] > 0.3
        assert 0.0 <= first_match["similarity_score"] <= 1.0

        # 7. Delete Chapter Memories
        del_res = client.delete(f"/api/v1/chapters/{chapter_id}/memories")
        assert del_res.status_code == 204

        # 8. Verify Status After Chapter Memory Deletion
        status_after = client.get(f"/api/v1/stories/{story_id}/memory-status").json()
        assert status_after["total_chunks"] == 0
        assert status_after["is_indexed"] is False

    def test_search_nonexistent_story_returns_404(self, client_with_chroma):
        client, _ = client_with_chroma
        res = client.post(
            "/api/v1/memory/search",
            json={"story_id": "nonexistent_story_id", "query": "Find memories"},
        )
        assert res.status_code == 404
        assert "not found" in res.json()["message"].lower()

    def test_index_nonexistent_chapter_returns_404(self, client_with_chroma):
        client, _ = client_with_chroma
        res = client.post("/api/v1/chapters/nonexistent_chapter_id/index")
        assert res.status_code == 404
        assert "not found" in res.json()["message"].lower()

    def test_search_empty_query_returns_422(self, client_with_chroma):
        client, _ = client_with_chroma
        res = client.post(
            "/api/v1/memory/search",
            json={"story_id": "story_123", "query": ""},
        )
        assert res.status_code == 422
