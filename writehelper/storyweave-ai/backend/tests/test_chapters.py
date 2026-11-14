"""
Tests for StoryWeave AI Chapter Management CRUD APIs.

Tests preservation of original manuscripts, status tracking, story isolation,
and pagination.
"""

from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.gemini_service import GeminiService
from app.database.repositories.dependencies import get_story_repository, get_chapter_repository


@pytest.fixture(autouse=True)
def clean_repositories():
    """Clear repositories before each test."""
    story_repo = get_story_repository()
    chapter_repo = get_chapter_repository()
    story_repo.clear()
    chapter_repo.clear()
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_story_id(client) -> str:
    """Helper fixture to create a valid parent story."""
    resp = client.post(
        "/api/v1/stories",
        json={"title": "The Whispering Woods", "genre": "Fantasy"},
    )
    return resp.json()["id"]


def test_create_chapter_success(client, sample_story_id):
    """Test successful chapter creation with original content preservation."""
    raw_manuscript = (
        "The wind howled through the twisted pines of Whispering Woods. "
        "Kael clutching his crystal dagger, stepped carefully over the frozen roots."
    )
    payload = {
        "title": "Into the Dark Woods",
        "chapter_number": 1,
        "content": raw_manuscript,
    }

    response = client.post(f"/api/v1/stories/{sample_story_id}/chapters", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert data["title"] == payload["title"]
    assert data["chapter_number"] == 1
    assert data["story_id"] == sample_story_id
    assert data["content"] == raw_manuscript
    assert data["original_content"] == raw_manuscript  # Preserved!
    assert data["word_count"] == len(raw_manuscript.split())
    assert data["processing_status"] == "pending"
    assert data["retry_count"] == 0
    assert data["error_message"] is None
    assert "id" in data


def test_create_chapter_invalid_story(client):
    """Test submitting chapter to a non-existent story returns 404."""
    payload = {
        "title": "Orphan Chapter",
        "chapter_number": 1,
        "content": "A story without a universe.",
    }
    response = client.post("/api/v1/stories/nonexistent_story_000/chapters", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert data["error"] == "StoryNotFoundException"


def test_create_duplicate_chapter_number(client, sample_story_id):
    """Test creating duplicate chapter sequence number returns 409 Conflict."""
    payload1 = {"title": "Chapter 1 First", "chapter_number": 1, "content": "Text 1"}
    payload2 = {"title": "Chapter 1 Duplicate", "chapter_number": 1, "content": "Text 2"}

    resp1 = client.post(f"/api/v1/stories/{sample_story_id}/chapters", json=payload1)
    assert resp1.status_code == 201

    resp2 = client.post(f"/api/v1/stories/{sample_story_id}/chapters", json=payload2)
    assert resp2.status_code == 409
    assert "already exists" in resp2.json()["message"]


def test_preserve_original_manuscript_on_update(client, sample_story_id):
    """CRITICAL: Test that editing working draft content NEVER overwrites original_content."""
    initial_manuscript = "This is the original rough first draft submitted by author."
    create_resp = client.post(
        f"/api/v1/stories/{sample_story_id}/chapters",
        json={"title": "Prologue", "chapter_number": 1, "content": initial_manuscript},
    )
    chapter_id = create_resp.json()["id"]

    revised_draft = "This is the polished second draft with extra narrative details added."
    update_resp = client.put(
        f"/api/v1/chapters/{chapter_id}",
        json={"title": "Prologue (Revised)", "content": revised_draft},
    )
    assert update_resp.status_code == 200
    updated_data = update_resp.json()

    # Working content is updated
    assert updated_data["content"] == revised_draft
    assert updated_data["title"] == "Prologue (Revised)"
    assert updated_data["word_count"] == len(revised_draft.split())

    # ORIGINAL CONTENT IS PRESERVED AND UNTOUCHED!
    assert updated_data["original_content"] == initial_manuscript

    # Verify fetching GET also has intact original_content
    get_resp = client.get(f"/api/v1/chapters/{chapter_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["original_content"] == initial_manuscript
    assert get_resp.json()["content"] == revised_draft


def test_list_chapters_ordered_by_sequence(client, sample_story_id):
    """Test listing chapters returns them sorted by chapter_number ascending."""
    # Create chapters out of order: 3, 1, 2
    client.post(
        f"/api/v1/stories/{sample_story_id}/chapters",
        json={"title": "Chapter 3", "chapter_number": 3, "content": "Text C"},
    )
    client.post(
        f"/api/v1/stories/{sample_story_id}/chapters",
        json={"title": "Chapter 1", "chapter_number": 1, "content": "Text A"},
    )
    client.post(
        f"/api/v1/stories/{sample_story_id}/chapters",
        json={"title": "Chapter 2", "chapter_number": 2, "content": "Text B"},
    )

    list_resp = client.get(f"/api/v1/stories/{sample_story_id}/chapters")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert data["total"] == 3
    numbers = [ch["chapter_number"] for ch in data["items"]]
    assert numbers == [1, 2, 3]


def test_list_chapters_pagination(client, sample_story_id):
    """Test chapter list pagination."""
    for i in range(1, 5):
        client.post(
            f"/api/v1/stories/{sample_story_id}/chapters",
            json={"title": f"Chapter {i}", "chapter_number": i, "content": f"Text {i}"},
        )

    # Page 1, limit 2
    resp1 = client.get(f"/api/v1/stories/{sample_story_id}/chapters?page=1&limit=2")
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert len(data1["items"]) == 2
    assert data1["total"] == 4
    assert data1["total_pages"] == 2
    assert data1["has_next"] is True

    # Page 2, limit 2
    resp2 = client.get(f"/api/v1/stories/{sample_story_id}/chapters?page=2&limit=2")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert len(data2["items"]) == 2
    assert data2["has_next"] is False
    assert data2["has_previous"] is True


def test_delete_chapter(client, sample_story_id):
    """Test deleting a chapter."""
    create_resp = client.post(
        f"/api/v1/stories/{sample_story_id}/chapters",
        json={"title": "Deleted Chap", "chapter_number": 1, "content": "Will be deleted"},
    )
    chapter_id = create_resp.json()["id"]

    del_resp = client.delete(f"/api/v1/chapters/{chapter_id}")
    assert del_resp.status_code == 204

    get_resp = client.get(f"/api/v1/chapters/{chapter_id}")
    assert get_resp.status_code == 404


def test_cascade_delete_story_removes_chapters(client, sample_story_id):
    """Test deleting a story also cleans up all of its chapters."""
    ch_resp = client.post(
        f"/api/v1/stories/{sample_story_id}/chapters",
        json={"title": "Chap In Story", "chapter_number": 1, "content": "Text"},
    )
    chapter_id = ch_resp.json()["id"]

    # Delete the story
    del_story_resp = client.delete(f"/api/v1/stories/{sample_story_id}")
    assert del_story_resp.status_code == 204

    # Chapter should no longer exist
    get_ch = client.get(f"/api/v1/chapters/{chapter_id}")
    assert get_ch.status_code == 404


def test_chapter_processing_status_endpoints(client, sample_story_id, sample_analysis_result):
    """Test /analyze and /retry-processing status lifecycle and report retrieval."""
    create_resp = client.post(
        f"/api/v1/stories/{sample_story_id}/chapters",
        json={"title": "Analysis Target", "chapter_number": 1, "content": "Some text to analyze"},
    )
    chapter_id = create_resp.json()["id"]

    # 1. Trigger analyze with mocked Gemini
    with patch.object(GeminiService, "analyze_chapter", new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = sample_analysis_result

        analyze_resp = client.post(f"/api/v1/chapters/{chapter_id}/analyze")
        assert analyze_resp.status_code == 200
        data = analyze_resp.json()
        assert data["chapter_id"] == chapter_id
        assert "analysis" in data
        assert len(data["analysis"]["characters"]) == 2

        # Verify chapter status is completed
        ch_resp = client.get(f"/api/v1/chapters/{chapter_id}")
        assert ch_resp.json()["processing_status"] == "completed"

        # 2. Retrieve analysis report via GET endpoint
        get_analysis_resp = client.get(f"/api/v1/chapters/{chapter_id}/analysis")
        assert get_analysis_resp.status_code == 200
        assert get_analysis_resp.json()["chapter_id"] == chapter_id

        # 3. Retry processing
        retry_resp = client.post(f"/api/v1/chapters/{chapter_id}/retry-processing")
        assert retry_resp.status_code == 200
        retry_data = retry_resp.json()
        assert retry_data["chapter_id"] == chapter_id

        # Verify chapter retry count incremented
        ch_retry_resp = client.get(f"/api/v1/chapters/{chapter_id}")
        assert ch_retry_resp.json()["retry_count"] == 1
