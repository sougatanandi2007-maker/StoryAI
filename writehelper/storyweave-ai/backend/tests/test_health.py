"""
Tests for StoryWeave AI Health Endpoints and Core App Startup.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database.mongodb import mongodb_manager
from app.database.chromadb import chromadb_manager


@pytest.fixture(scope="session", autouse=True)
def init_test_environment():
    """Ensure database managers are connected during test session."""
    mongodb_manager.connect()
    chromadb_manager.connect()
    yield
    mongodb_manager.disconnect()
    chromadb_manager.disconnect()


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_app_health(client):
    """Test the base application health check."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "StoryWeave" in data["application"]
    assert "version" in data
    assert "timestamp" in data


def test_database_health(client):
    """Test database health check endpoint."""
    response = client.get("/health/database")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "databases" in data
    assert "mongodb" in data["databases"]
    assert "chromadb" in data["databases"]
    # ChromaDB should be healthy since local storage initialized
    assert data["databases"]["chromadb"]["status"] == "healthy"
    # MongoDB will be not_configured until credentials are provided
    assert data["databases"]["mongodb"]["status"] in ("not_configured", "healthy", "disconnected")


def test_ai_health(client):
    """Test AI services health check endpoint."""
    response = client.get("/health/ai")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert "gemini" in data["services"]
    assert "embedding" in data["services"]
    assert data["services"]["gemini"]["status"] in ("not_configured", "healthy")
    assert data["services"]["embedding"]["status"] == "available"


def test_docs_and_openapi(client):
    """Test Swagger UI and OpenAPI documentation availability."""
    docs_response = client.get("/docs")
    assert docs_response.status_code == 200

    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200
    schema = openapi_response.json()
    assert schema["info"]["title"] == "StoryWeave AI"
    assert "/api/v1/stories" in schema["paths"]
    assert "/api/v1/chapters/{chapter_id}" in schema["paths"]
    assert "/health" in schema["paths"]
