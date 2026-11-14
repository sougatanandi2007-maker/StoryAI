"""
Tests for StoryWeave AI Story Management CRUD APIs.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database.repositories.dependencies import get_story_repository, get_chapter_repository


@pytest.fixture(autouse=True)
def clean_repositories():
    """Clear in-memory repositories before each test."""
    story_repo = get_story_repository()
    chapter_repo = get_chapter_repository()
    story_repo.clear()
    chapter_repo.clear()
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_create_story_success(client):
    """Test successful story creation."""
    payload = {
        "title": "Chronicles of Eldoria",
        "synopsis": "A high fantasy realm where magic is fueled by ancient celestial crystals.",
        "genre": "Epic Fantasy",
        "world_rules_overview": "Magic requires physical crystal catalysts; no mortal can cast raw ether.",
        "metadata": {"target_audience": "Adult", "tone": "Dark & Gritty"},
    }
    response = client.post("/api/v1/stories", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == payload["title"]
    assert data["synopsis"] == payload["synopsis"]
    assert data["genre"] == payload["genre"]
    assert data["world_rules_overview"] == payload["world_rules_overview"]
    assert data["metadata"]["tone"] == "Dark & Gritty"
    assert data["chapter_count"] == 0
    assert "id" in data
    assert "user_id" in data
    assert "created_at" in data
    assert "updated_at" in data


def test_create_story_validation_error(client):
    """Test validation errors on empty or missing fields."""
    # Missing required 'title'
    response = client.post("/api/v1/stories", json={"genre": "Sci-Fi"})
    assert response.status_code == 422

    # Empty title string
    response = client.post("/api/v1/stories", json={"title": "   "})
    # Pydantic min_length=1 catches or service strips
    assert response.status_code in (400, 422)


def test_get_story_by_id(client):
    """Test retrieving an existing story."""
    create_resp = client.post(
        "/api/v1/stories",
        json={"title": "Cyber City 2099", "genre": "Cyberpunk"},
    )
    story_id = create_resp.json()["id"]

    get_resp = client.get(f"/api/v1/stories/{story_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["id"] == story_id
    assert data["title"] == "Cyber City 2099"


def test_get_nonexistent_story(client):
    """Test retrieving a story that does not exist returns 404."""
    response = client.get("/api/v1/stories/nonexistent_story_123")
    assert response.status_code == 404
    data = response.json()
    assert data["error"] == "StoryNotFoundException"
    assert "nonexistent_story_123" in data["message"]


def test_list_stories_with_pagination(client):
    """Test pagination when listing stories."""
    # Create 5 stories
    for i in range(1, 6):
        client.post(
            "/api/v1/stories",
            json={"title": f"Story Universe {i}", "genre": "Fiction"},
        )

    # Page 1, limit 2
    resp1 = client.get("/api/v1/stories?page=1&limit=2")
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert len(data1["items"]) == 2
    assert data1["total"] == 5
    assert data1["page"] == 1
    assert data1["limit"] == 2
    assert data1["total_pages"] == 3
    assert data1["has_next"] is True
    assert data1["has_previous"] is False

    # Page 3, limit 2 (last page, 1 item)
    resp3 = client.get("/api/v1/stories?page=3&limit=2")
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert len(data3["items"]) == 1
    assert data3["has_next"] is False
    assert data3["has_previous"] is True


def test_update_story(client):
    """Test updating story fields."""
    create_resp = client.post(
        "/api/v1/stories",
        json={"title": "Draft Title", "synopsis": "Initial plot."},
    )
    story_id = create_resp.json()["id"]

    update_payload = {
        "title": "Final Polished Title",
        "synopsis": "Expanded premise with rich twists.",
        "genre": "Mystery Thriller",
    }
    update_resp = client.put(f"/api/v1/stories/{story_id}", json=update_payload)
    assert update_resp.status_code == 200
    data = update_resp.json()
    assert data["title"] == "Final Polished Title"
    assert data["synopsis"] == "Expanded premise with rich twists."
    assert data["genre"] == "Mystery Thriller"


def test_update_nonexistent_story(client):
    """Test updating non-existent story returns 404."""
    response = client.put(
        "/api/v1/stories/nonexistent_story_999",
        json={"title": "Ghost Story"},
    )
    assert response.status_code == 404


def test_delete_story_success(client):
    """Test deleting a story."""
    create_resp = client.post(
        "/api/v1/stories",
        json={"title": "Temporary Universe"},
    )
    story_id = create_resp.json()["id"]

    del_resp = client.delete(f"/api/v1/stories/{story_id}")
    assert del_resp.status_code == 204

    # Subsequent GET returns 404
    get_resp = client.get(f"/api/v1/stories/{story_id}")
    assert get_resp.status_code == 404


def test_delete_nonexistent_story(client):
    """Test deleting non-existent story returns 404."""
    del_resp = client.delete("/api/v1/stories/nonexistent_story_888")
    assert del_resp.status_code == 404
