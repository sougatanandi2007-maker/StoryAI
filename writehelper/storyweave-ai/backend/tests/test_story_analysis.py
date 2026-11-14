"""
Tests for StoryWeave AI Phase 3 - Gemini AI Story Analysis & Entity Extraction.

Covers:
- Successful analysis & entity extraction (Characters, Relationships, Events, Locations, Facts, Rules, Timeline)
- Evidence validation and Explicit vs Inferred fact classification
- Malformed AI responses and error recovery
- Missing API credentials handling
- Status lifecycle (pending -> processing -> completed / failed)
- Manuscript preservation guarantee
- Idempotent analysis reports (no duplicates on retry)
- Real Gemini Flash API call (live integration test)
"""

import json
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_settings
from app.services.gemini_service import GeminiService
from app.schemas.chapter import ProcessingStatus
from app.schemas.analysis import ChapterAnalysisResult, FactType
from app.core.exceptions import (
    GeminiNotConfiguredError,
    GeminiAPIError,
    MalformedAnalysisResponseError,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_story_and_chapter(client):
    """Fixture providing a valid story and chapter for analysis testing."""
    story_resp = client.post(
        "/api/v1/stories",
        json={
            "title": "The Crystal Prophecy",
            "synopsis": "A world where celestial crystals channel magic.",
            "genre": "High Fantasy",
            "world_rules_overview": "Ether cannot be cast without crystal catalysts.",
        },
    )
    story_id = story_resp.json()["id"]

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

    chap_resp = client.post(
        f"/api/v1/stories/{story_id}/chapters",
        json={
            "title": "Into the Dark Woods",
            "chapter_number": 1,
            "content": manuscript,
        },
    )
    chapter_id = chap_resp.json()["id"]

    return {
        "story_id": story_id,
        "chapter_id": chapter_id,
        "manuscript": manuscript,
    }


def test_successful_analysis_and_extraction(client, test_story_and_chapter, sample_analysis_result):
    """Test full extraction pipeline with structured JSON result."""
    chapter_id = test_story_and_chapter["chapter_id"]

    with patch.object(GeminiService, "analyze_chapter", new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = sample_analysis_result

        response = client.post(f"/api/v1/chapters/{chapter_id}/analyze")
        assert response.status_code == 200
        data = response.json()

        assert data["chapter_id"] == chapter_id
        assert data["story_id"] == test_story_and_chapter["story_id"]

        analysis = data["analysis"]
        # 1. Summary
        assert "Kael arrives at the Whispering Woods" in analysis["chapter_summary"]["summary"]
        assert len(analysis["chapter_summary"]["key_takeaways"]) >= 3
        assert analysis["chapter_summary"]["pov_character"] == "Kael"

        # 2. Characters
        assert len(analysis["characters"]) == 2
        kael = next(c for c in analysis["characters"] if "Kael" in c["name"])
        assert "Determined" in kael["personality_traits"]
        assert len(kael["evidence"]) > 0

        # 3. Relationships
        assert len(analysis["relationships"]) == 1
        rel = analysis["relationships"][0]
        assert rel["relationship_type"] == "ally"
        assert len(rel["evidence"]) > 0

        # 4. Events
        assert len(analysis["events"]) == 1
        event = analysis["events"][0]
        assert event["title"] == "Ambush in the Whispering Pines"
        assert len(event["evidence"]) > 0

        # 5. Locations
        assert len(analysis["locations"]) == 1
        assert analysis["locations"][0]["name"] == "Whispering Woods"

        # 6. Facts (Explicit vs Inferred verification)
        assert len(analysis["story_facts"]) == 2
        explicit_fact = next(f for f in analysis["story_facts"] if f["fact_type"] == "explicit")
        inferred_fact = next(f for f in analysis["story_facts"] if f["fact_type"] == "inferred")
        assert explicit_fact["confidence"] == 1.0
        assert inferred_fact["confidence"] < 1.0
        assert len(explicit_fact["evidence"]) > 0
        assert len(inferred_fact["evidence"]) > 0

        # 7. World rules
        assert len(analysis["world_rules"]) == 1
        assert "crystal catalysts" in analysis["world_rules"][0]["rule"]

        # 8. Timeline
        assert len(analysis["timeline"]) == 1
        assert "Blood Eclipse" in analysis["timeline"][0]["time_anchor"]

        # Verify chapter status is marked COMPLETED
        ch_resp = client.get(f"/api/v1/chapters/{chapter_id}")
        assert ch_resp.json()["processing_status"] == "completed"


def test_get_chapter_analysis_report(client, test_story_and_chapter, sample_analysis_result):
    """Test retrieving an existing analysis report via GET."""
    chapter_id = test_story_and_chapter["chapter_id"]

    # Before analysis: 404
    get_before = client.get(f"/api/v1/chapters/{chapter_id}/analysis")
    assert get_before.status_code == 404

    # Run analysis
    with patch.object(GeminiService, "analyze_chapter", new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = sample_analysis_result
        client.post(f"/api/v1/chapters/{chapter_id}/analyze")

    # After analysis: 200 with report
    get_after = client.get(f"/api/v1/chapters/{chapter_id}/analysis")
    assert get_after.status_code == 200
    report = get_after.json()
    assert report["chapter_id"] == chapter_id
    assert report["analysis"]["chapter_summary"]["pov_character"] == "Kael"


def test_idempotent_analysis_no_duplicates(client, test_story_and_chapter, sample_analysis_result):
    """Test that multiple analyses on the same chapter update the report without duplicates."""
    chapter_id = test_story_and_chapter["chapter_id"]

    with patch.object(GeminiService, "analyze_chapter", new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = sample_analysis_result

        # Run 1
        resp1 = client.post(f"/api/v1/chapters/{chapter_id}/analyze")
        report_id_1 = resp1.json()["id"]

        # Run 2 (retry)
        resp2 = client.post(f"/api/v1/chapters/{chapter_id}/retry-processing")
        report_id_2 = resp2.json()["id"]

        # Must update the existing report document, not create duplicates
        assert report_id_1 == report_id_2


def test_manuscript_preservation_during_analysis(client, test_story_and_chapter, sample_analysis_result):
    """Test that original manuscript and working draft remain intact after analysis."""
    chapter_id = test_story_and_chapter["chapter_id"]
    original_text = test_story_and_chapter["manuscript"]

    with patch.object(GeminiService, "analyze_chapter", new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = sample_analysis_result
        client.post(f"/api/v1/chapters/{chapter_id}/analyze")

    ch_resp = client.get(f"/api/v1/chapters/{chapter_id}")
    data = ch_resp.json()
    assert data["original_content"] == original_text
    assert data["content"] == original_text
    assert data["processing_status"] == "completed"


def test_malformed_ai_response_handling(client, test_story_and_chapter):
    """Test handling of malformed or invalid JSON from Gemini."""
    chapter_id = test_story_and_chapter["chapter_id"]

    with patch.object(GeminiService, "analyze_chapter", new_callable=AsyncMock) as mock_analyze:
        mock_analyze.side_effect = MalformedAnalysisResponseError(
            message="Invalid JSON structure from model",
            raw_output="I am unable to output JSON for this text.",
        )

        response = client.post(f"/api/v1/chapters/{chapter_id}/analyze")
        assert response.status_code == 502
        data = response.json()
        assert "MalformedAnalysisResponseError" in data["error"]

        # Chapter status must be marked FAILED with recorded error message
        ch_resp = client.get(f"/api/v1/chapters/{chapter_id}")
        ch_data = ch_resp.json()
        assert ch_data["processing_status"] == "failed"
        assert "Invalid JSON structure" in ch_data["error_message"]


def test_missing_credentials_handling(client, test_story_and_chapter):
    """Test error handling when GEMINI_API_KEY is not configured."""
    chapter_id = test_story_and_chapter["chapter_id"]

    with patch.object(GeminiService, "analyze_chapter", new_callable=AsyncMock) as mock_analyze:
        mock_analyze.side_effect = GeminiNotConfiguredError()

        response = client.post(f"/api/v1/chapters/{chapter_id}/analyze")
        assert response.status_code == 503
        data = response.json()
        assert "GeminiNotConfiguredError" in data["error"]

        # Chapter status must be marked FAILED
        ch_resp = client.get(f"/api/v1/chapters/{chapter_id}")
        assert ch_resp.json()["processing_status"] == "failed"


def test_api_failure_and_retry_recovery(client, test_story_and_chapter, sample_analysis_result):
    """Test recovering from a failed analysis attempt via retry-processing."""
    chapter_id = test_story_and_chapter["chapter_id"]

    # 1. First run fails with API error
    with patch.object(GeminiService, "analyze_chapter", new_callable=AsyncMock) as mock_fail:
        mock_fail.side_effect = GeminiAPIError("Gemini rate limit exceeded.")
        client.post(f"/api/v1/chapters/{chapter_id}/analyze")

    ch_after_fail = client.get(f"/api/v1/chapters/{chapter_id}").json()
    assert ch_after_fail["processing_status"] == "failed"
    assert "rate limit" in ch_after_fail["error_message"]

    # 2. Retry succeeds
    with patch.object(GeminiService, "analyze_chapter", new_callable=AsyncMock) as mock_success:
        mock_success.return_value = sample_analysis_result
        retry_resp = client.post(f"/api/v1/chapters/{chapter_id}/retry-processing")
        assert retry_resp.status_code == 200

    ch_after_retry = client.get(f"/api/v1/chapters/{chapter_id}").json()
    assert ch_after_retry["processing_status"] == "completed"
    assert ch_after_retry["error_message"] is None
    assert ch_after_retry["retry_count"] == 1


@pytest.mark.live_api
@pytest.mark.skipif(
    not get_settings().is_gemini_configured,
    reason="Live Gemini API key not configured.",
)
def test_live_gemini_analysis_integration(client, test_story_and_chapter):
    """
    Live end-to-end integration test calling the real Google Gemini API.
    Only runs when GEMINI_API_KEY is configured in the environment.
    """
    chapter_id = test_story_and_chapter["chapter_id"]

    # Make the actual real Gemini call
    response = client.post(f"/api/v1/chapters/{chapter_id}/analyze")
    assert response.status_code == 200
    data = response.json()

    assert data["chapter_id"] == chapter_id
    analysis = data["analysis"]

    # Verify real extracted entities
    assert len(analysis["chapter_summary"]["summary"]) > 50
    assert len(analysis["characters"]) >= 1

    # Verify characters have non-empty evidence
    for char in analysis["characters"]:
        assert len(char["name"]) > 0
        assert len(char["evidence"]) > 0

    # Verify status in database
    ch_resp = client.get(f"/api/v1/chapters/{chapter_id}").json()
    assert ch_resp["processing_status"] == "completed"
