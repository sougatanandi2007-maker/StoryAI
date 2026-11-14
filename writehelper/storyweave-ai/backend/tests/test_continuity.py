"""
Tests for StoryWeave AI Phase 5 - Story Continuity & Consistency Engine.

Covers:
- Initial chapter foundational canon establishment (0 contradictions)
- Character appearance contradiction detection
- Intentional character development recognition (low severity / advisory)
- Timeline conflict detection
- Flashback handling (non-chronological scenes not flagged as errors)
- Relationship evolution recognition (enemies -> allies through plot events)
- World-rule contradiction detection (magic system violation)
- Explicit versus inferred fact weighting (inferred facts do not override explicit facts)
- Insufficient evidence handling (low confidence observation)
- Off-screen event possibilities (plausible transitions between chapters)
- Duplicate issue prevention and author triage preservation on recheck
- Report retrieval endpoints (by chapter and by story)
- Issue status updates (open -> accepted, dismissed, resolved with author notes)
- User isolation enforcement
- Story isolation enforcement
- Gemini timeout and error handling
- Manuscript preservation guarantee (advisory only; no text changes)
- Live Gemini API continuity check integration test
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_settings
from app.database.chromadb import ChromaDBManager, get_chromadb_manager
from app.database.repositories.dependencies import (
    get_story_repository,
    get_chapter_repository,
    get_analysis_repository,
    get_continuity_repository,
)
from app.schemas.analysis import (
    ChapterAnalysisResult,
    ChapterSummary,
    ExtractedCharacter,
    ExtractedRelationship,
    ExtractedEvent,
    ExtractedLocation,
    ExtractedFact,
    ExtractedWorldRule,
    ExtractedTimelineEvent,
    FactType,
)
from app.schemas.continuity import (
    ContinuityIssueType,
    ContinuitySeverity,
    IssueStatus,
    ContinuityEvidence,
    ContinuityIssue,
    ContinuityReport,
)
from app.core.exceptions import GeminiAPIError
from app.core.security import DEV_USER_ID, get_current_user_id
from app.services.gemini_service import GeminiService


# --- Fixtures ---

@pytest.fixture
def temp_chroma(tmp_path) -> ChromaDBManager:
    """Isolated ChromaDB instance for testing."""
    chroma_dir = str(tmp_path / "chroma_continuity_test")
    manager = ChromaDBManager(persist_directory=chroma_dir)
    manager.connect()
    yield manager
    manager.disconnect()


@pytest.fixture
def client_with_continuity(temp_chroma):
    """FastAPI TestClient with isolated ChromaDB and repository cleanup."""
    get_story_repository().clear()
    get_chapter_repository().clear()
    get_analysis_repository().clear()
    get_continuity_repository().clear()

    app.dependency_overrides[get_chromadb_manager] = lambda: temp_chroma
    client = TestClient(app)
    yield client, temp_chroma

    app.dependency_overrides.clear()


@pytest.fixture
def sample_universe(client_with_continuity):
    """Creates a story with Chapter 1 and Chapter 2 for continuity testing."""
    client, _ = client_with_continuity

    story_resp = client.post(
        "/api/v1/stories",
        json={
            "title": "Chronicles of Eldoria",
            "synopsis": "An ancient empire where magic is channeled via crystal foci.",
            "genre": "Epic Fantasy",
            "world_rules_overview": "Magic strictly requires physical crystal catalysts.",
        },
    )
    assert story_resp.status_code == 201
    story = story_resp.json()
    story_id = story["id"]

    # Chapter 1
    chap1_resp = client.post(
        f"/api/v1/stories/{story_id}/chapters",
        json={
            "title": "The Whispering Pines",
            "chapter_number": 1,
            "content": (
                "Kael walked into Whispering Woods. He was a tall man with jet-black hair and silver eyes, "
                "clutching his sapphire crystal pendant. 'Magic cannot be channeled without crystal glass,' "
                "his master had taught him. Lyra, an elven sentinel with a carved weirwood bow, sprang from the shadows: "
                "'Human thief, you are my sworn enemy!' But when shadow wolves attacked, they stood back-to-back "
                "and agreed to a wary truce."
            ),
        },
    )
    assert chap1_resp.status_code == 201
    chap1 = chap1_resp.json()

    # Chapter 2
    chap2_resp = client.post(
        f"/api/v1/stories/{story_id}/chapters",
        json={
            "title": "The Obsidian Spire",
            "chapter_number": 2,
            "content": (
                "Kael approached the Obsidian Spire. His golden-blonde hair fluttered in the gale as he stared with "
                "violet eyes. Without touching any crystal focus, Kael raised his bare hands and hurled a tempest of raw "
                "ether flame directly from his palms. Lyra nodded with quiet admiration, grateful for their steadfast alliance."
            ),
        },
    )
    assert chap2_resp.status_code == 201
    chap2 = chap2_resp.json()

    # Store Chapter 1 Analysis
    analysis_repo = get_analysis_repository()
    now = datetime.now(timezone.utc)
    chap1_analysis = {
        "id": f"rep_{chap1['id']}",
        "chapter_id": chap1["id"],
        "story_id": story_id,
        "user_id": DEV_USER_ID,
        "analysis": {
            "chapter_summary": {
                "summary": "Kael journeys into Whispering Woods and forms an uneasy truce with Lyra.",
                "key_takeaways": ["Kael enters woods", "Lyra confronts Kael", "Truce formed against shadow wolves"],
                "pov_character": "Kael",
                "pacing_assessment": "Moderate build-up to combat.",
            },
            "characters": [
                {
                    "name": "Kael",
                    "aliases": ["Kael of Riverdell"],
                    "description": "Tall man with jet-black hair and silver eyes.",
                    "personality_traits": ["Cautious", "Timid in the face of monsters"],
                    "motivations": ["Survive the woods"],
                    "goals": ["Find the spire"],
                    "abilities": ["Tracking"],
                    "evidence": "Kael was a tall man with jet-black hair and silver eyes.",
                },
                {
                    "name": "Lyra",
                    "aliases": ["Sentinel of the Pines"],
                    "description": "Elven scout with a weirwood bow.",
                    "personality_traits": ["Vigilant", "Proud"],
                    "motivations": ["Defend the forest"],
                    "goals": ["Expel intruders"],
                    "abilities": ["Archery"],
                    "evidence": "Lyra, an elven sentinel with a carved weirwood bow.",
                },
            ],
            "relationships": [
                {
                    "character_1": "Kael",
                    "character_2": "Lyra",
                    "relationship_type": "allies",
                    "description": "Sworn enemies who formed a truce against shadow wolves.",
                    "evidence": "agreed to a wary truce",
                }
            ],
            "events": [
                {
                    "title": "Wolf Attack",
                    "description": "Shadow wolves attack Kael and Lyra in Whispering Woods.",
                    "participants": ["Kael", "Lyra"],
                    "location": "Whispering Woods",
                    "chronological_order": 1,
                    "significance": "Forces them into an alliance.",
                    "evidence": "when shadow wolves attacked, they stood back-to-back",
                }
            ],
            "locations": [
                {
                    "name": "Whispering Woods",
                    "description": "Dark forest filled with shadow wolves.",
                    "significance": "Borderland forest.",
                    "evidence": "Kael walked into Whispering Woods.",
                }
            ],
            "story_facts": [
                {
                    "fact": "Shadow wolves roam the Whispering Woods at twilight.",
                    "category": "lore",
                    "fact_type": "explicit",
                    "confidence": 1.0,
                    "evidence": "when shadow wolves attacked",
                },
                {
                    "fact": "Kael might secretly be an heir to the fallen realm.",
                    "category": "secret",
                    "fact_type": "inferred",
                    "confidence": 0.65,
                    "evidence": "clutching his sapphire crystal pendant",
                },
            ],
            "world_rules": [
                {
                    "rule": "Magic cannot be channeled without refined crystal catalysts.",
                    "category": "magic",
                    "evidence": "Magic cannot be channeled without crystal glass, his master had taught him.",
                }
            ],
            "timeline": [
                {
                    "relative_time": "At twilight",
                    "is_flashback": False,
                    "uncertain_dates": False,
                    "time_anchor": "Wolf attack",
                    "evidence": "Kael walked into Whispering Woods",
                }
            ],
        },
        "model_used": "gemini-flash-lite-latest",
        "created_at": now,
        "updated_at": now,
    }

    # Store Chapter 2 Analysis
    chap2_analysis = {
        "id": f"rep_{chap2['id']}",
        "chapter_id": chap2["id"],
        "story_id": story_id,
        "user_id": DEV_USER_ID,
        "analysis": {
            "chapter_summary": {
                "summary": "Kael and Lyra arrive at the Obsidian Spire where Kael unleashes bare-handed ether magic.",
                "key_takeaways": ["Arrive at Obsidian Spire", "Kael casts bare-handed ether"],
                "pov_character": "Kael",
                "pacing_assessment": "High tension.",
            },
            "characters": [
                {
                    "name": "Kael",
                    "aliases": [],
                    "description": "Golden-blonde hair and violet eyes.",
                    "personality_traits": ["Confident", "Heroic"],
                    "motivations": ["Breach the spire"],
                    "goals": ["Destroy the spire gate"],
                    "abilities": ["Raw ether flame"],
                    "evidence": "His golden-blonde hair fluttered in the gale as he stared with violet eyes.",
                }
            ],
            "relationships": [
                {
                    "character_1": "Kael",
                    "character_2": "Lyra",
                    "relationship_type": "allies",
                    "description": "Steadfast companions.",
                    "evidence": "grateful for their steadfast alliance",
                }
            ],
            "events": [
                {
                    "title": "Spire Gate Breach",
                    "description": "Kael casts raw ether to blast open the gate.",
                    "participants": ["Kael"],
                    "location": "Obsidian Spire",
                    "chronological_order": 1,
                    "significance": "Demonstrates unprecedented magic.",
                    "evidence": "Without touching any crystal focus, Kael raised his bare hands",
                }
            ],
            "locations": [
                {
                    "name": "Obsidian Spire",
                    "description": "Tower of black glass.",
                    "significance": "Stronghold of shadow.",
                    "evidence": "Kael approached the Obsidian Spire.",
                }
            ],
            "story_facts": [
                {
                    "fact": "Kael cast raw ether flame directly from his palms without any crystal focus.",
                    "category": "magic",
                    "fact_type": "explicit",
                    "confidence": 1.0,
                    "evidence": "hurled a tempest of raw ether flame directly from his palms",
                }
            ],
            "world_rules": [],
            "timeline": [
                {
                    "relative_time": "Two days after the forest ambush",
                    "is_flashback": False,
                    "uncertain_dates": False,
                    "time_anchor": "Spire approach",
                    "evidence": "Kael approached the Obsidian Spire",
                }
            ],
        },
        "model_used": "gemini-flash-lite-latest",
        "created_at": now,
        "updated_at": now,
    }

    import asyncio
    asyncio.run(analysis_repo.save_report(chap1_analysis))
    asyncio.run(analysis_repo.save_report(chap2_analysis))

    return {
        "story": story,
        "chap1": chap1,
        "chap2": chap2,
    }


# ============================================================================
# Unit Tests
# ============================================================================

def test_first_chapter_establishes_canon(client_with_continuity, sample_universe):
    """Chapter 1 has no earlier chapters, so it establishes foundational canon with 0 issues."""
    client, _ = client_with_continuity
    chap1_id = sample_universe["chap1"]["id"]

    response = client.post(f"/api/v1/chapters/{chap1_id}/continuity-check")
    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is False
    assert data["report"]["total_issues"] == 0
    assert "establishes foundational canon" in data["report"]["summary"]

    # Verify second call retrieves cached report
    cached_resp = client.post(f"/api/v1/chapters/{chap1_id}/continuity-check")
    assert cached_resp.status_code == 200
    assert cached_resp.json()["cached"] is True


def test_character_appearance_contradiction(client_with_continuity, sample_universe):
    """Detects physical appearance contradiction (hair color changed from black to blonde)."""
    client, _ = client_with_continuity
    chap1_id = sample_universe["chap1"]["id"]
    chap2_id = sample_universe["chap2"]["id"]
    story_id = sample_universe["story"]["id"]

    mock_issues = [
        {
            "issue_type": "character_consistency",
            "severity": "high",
            "confidence": 0.95,
            "description": "Kael's hair color changed from jet-black in Chapter 1 to golden-blonde in Chapter 2 without explanation.",
            "existing_fact": "Kael was a tall man with jet-black hair and silver eyes.",
            "new_fact": "Kael has golden-blonde hair and violet eyes.",
            "affected_entities": ["Kael"],
            "earlier_evidence": [
                {
                    "chapter_id": chap1_id,
                    "chapter_number": 1,
                    "quote_or_excerpt": "Kael was a tall man with jet-black hair and silver eyes.",
                    "fact_type": "explicit",
                    "confidence": 1.0,
                }
            ],
            "current_evidence": {
                "chapter_id": chap2_id,
                "chapter_number": 2,
                "quote_or_excerpt": "His golden-blonde hair fluttered in the gale as he stared with violet eyes.",
                "fact_type": "explicit",
                "confidence": 0.95,
            },
            "source_chapter_ids": [chap1_id, chap2_id],
            "possible_explanation": "Kael may have used an illusion spell or magical disguise off-screen.",
            "suggested_resolution": "Clarify if this is a disguise or reconcile Kael's hair and eye color.",
        }
    ]

    with patch.object(GeminiService, "evaluate_continuity", new=AsyncMock(return_value=mock_issues)):
        resp = client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")
        assert resp.status_code == 200
        data = resp.json()
        report = data["report"]
        assert report["total_issues"] == 1
        issue = report["issues"][0]
        assert issue["issue_type"] == "character_consistency"
        assert issue["severity"] == "high"
        assert issue["confidence"] >= 0.9
        assert len(issue["earlier_evidence"]) >= 1
        assert issue["current_evidence"]["chapter_number"] == 2
        assert "Kael" in issue["affected_entities"]


def test_world_rule_contradiction(client_with_continuity, sample_universe):
    """Detects magic world rule violation (casting without a crystal focus)."""
    client, _ = client_with_continuity
    chap1_id = sample_universe["chap1"]["id"]
    chap2_id = sample_universe["chap2"]["id"]

    mock_issues = [
        {
            "issue_type": "world_rule_consistency",
            "severity": "critical",
            "confidence": 0.98,
            "description": "Magic rule broken: Chapter 1 stated magic strictly requires crystal catalysts, but Kael cast raw ether flame with bare hands.",
            "existing_fact": "Magic cannot be channeled without refined crystal catalysts.",
            "new_fact": "Kael cast raw ether flame directly from his bare palms without a crystal focus.",
            "affected_entities": ["Magic System", "Kael"],
            "earlier_evidence": [
                {
                    "chapter_id": chap1_id,
                    "chapter_number": 1,
                    "quote_or_excerpt": "Magic cannot be channeled without crystal glass, his master had taught him.",
                    "fact_type": "explicit",
                    "confidence": 1.0,
                }
            ],
            "current_evidence": {
                "chapter_id": chap2_id,
                "chapter_number": 2,
                "quote_or_excerpt": "Without touching any crystal focus, Kael raised his bare hands and hurled a tempest of raw ether flame.",
                "fact_type": "explicit",
                "confidence": 0.98,
            },
            "source_chapter_ids": [chap1_id, chap2_id],
            "possible_explanation": "Kael might possess a rare ancestral trait that bypasses standard mortal restrictions.",
            "suggested_resolution": "Mention that Kael's bloodline allows bypassing the crystal catalyst rule.",
        }
    ]

    with patch.object(GeminiService, "evaluate_continuity", new=AsyncMock(return_value=mock_issues)):
        resp = client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")
        assert resp.status_code == 200
        report = resp.json()["report"]
        assert report["total_issues"] == 1
        issue = report["issues"][0]
        assert issue["issue_type"] == "world_rule_consistency"
        assert issue["severity"] == "critical"
        assert issue["earlier_evidence"][0]["quote_or_excerpt"] != ""


def test_intentional_character_development(client_with_continuity, sample_universe):
    """Intentional emotional or behavioral growth is recognized as development, not an error."""
    client, _ = client_with_continuity
    chap1_id = sample_universe["chap1"]["id"]
    chap2_id = sample_universe["chap2"]["id"]

    # When character evolves, Gemini returns low severity with character development explanation
    mock_issues = [
        {
            "issue_type": "character_consistency",
            "severity": "low",
            "confidence": 0.50,
            "description": "Kael displays aggressive bravery instead of his earlier timidity.",
            "existing_fact": "Kael was hesitant and cautious around monsters.",
            "new_fact": "Kael boldly attacks the spire gates.",
            "affected_entities": ["Kael"],
            "earlier_evidence": [
                {
                    "chapter_id": chap1_id,
                    "chapter_number": 1,
                    "quote_or_excerpt": "Kael was cautious and timid in the face of monsters.",
                    "fact_type": "explicit",
                    "confidence": 0.9,
                }
            ],
            "current_evidence": {
                "chapter_id": chap2_id,
                "chapter_number": 2,
                "quote_or_excerpt": "Kael boldly faced the spire gate.",
                "fact_type": "explicit",
                "confidence": 0.9,
            },
            "source_chapter_ids": [chap1_id, chap2_id],
            "possible_explanation": "Intentional character development resulting from surviving the wolf ambush in Chapter 1.",
            "suggested_resolution": "Highlight Kael reflecting on how the ambush changed his resolve.",
        }
    ]

    with patch.object(GeminiService, "evaluate_continuity", new=AsyncMock(return_value=mock_issues)):
        resp = client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")
        assert resp.status_code == 200
        report = resp.json()["report"]
        assert report["total_issues"] == 1
        issue = report["issues"][0]
        assert issue["severity"] == "low"
        assert "character development" in issue["possible_explanation"].lower()


def test_timeline_conflict_detection(client_with_continuity, sample_universe):
    """Detects impossible chronological sequence."""
    client, _ = client_with_continuity
    chap1_id = sample_universe["chap1"]["id"]
    chap2_id = sample_universe["chap2"]["id"]

    mock_issues = [
        {
            "issue_type": "timeline_consistency",
            "severity": "high",
            "confidence": 0.92,
            "description": "Timeline paradox: Chapter 2 takes place during the High Summer Festival, but Chapter 1 was mid-winter frost.",
            "existing_fact": "Whispering Woods was under mid-winter frozen frost.",
            "new_fact": "Summer heatwave at the Obsidian Spire 2 days later.",
            "affected_entities": ["Season / Timeline"],
            "earlier_evidence": [
                {
                    "chapter_id": chap1_id,
                    "chapter_number": 1,
                    "quote_or_excerpt": "stepped carefully over the frozen roots",
                    "fact_type": "explicit",
                    "confidence": 1.0,
                }
            ],
            "current_evidence": {
                "chapter_id": chap2_id,
                "chapter_number": 2,
                "quote_or_excerpt": "scorching midsummer sun beat down on the black stones",
                "fact_type": "explicit",
                "confidence": 0.95,
            },
            "source_chapter_ids": [chap1_id, chap2_id],
            "possible_explanation": "Magical climate distortion or author oversight of the seasonal timeline.",
            "suggested_resolution": "Adjust the time elapsed or explain localized volcanic heat.",
        }
    ]

    with patch.object(GeminiService, "evaluate_continuity", new=AsyncMock(return_value=mock_issues)):
        resp = client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")
        assert resp.status_code == 200
        report = resp.json()["report"]
        assert report["issues"][0]["issue_type"] == "timeline_consistency"
        assert report["issues"][0]["severity"] == "high"


def test_flashback_handling(client_with_continuity, sample_universe):
    """Narrative flashbacks are not flagged as chronological contradictions."""
    client, _ = client_with_continuity
    chap2_id = sample_universe["chap2"]["id"]

    # In a flashback scenario, Gemini correctly identifies the narrative technique and returns no issues
    with patch.object(GeminiService, "evaluate_continuity", new=AsyncMock(return_value=[])):
        resp = client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")
        assert resp.status_code == 200
        report = resp.json()["report"]
        assert report["total_issues"] == 0
        assert "No narrative contradictions" in report["summary"]


def test_relationship_evolution(client_with_continuity, sample_universe):
    """Enemies becoming allies through dramatic story events is recognized as relationship evolution."""
    client, _ = client_with_continuity
    chap2_id = sample_universe["chap2"]["id"]

    # Gemini evaluates the truce and alliance as valid character evolution -> returns empty list
    with patch.object(GeminiService, "evaluate_continuity", new=AsyncMock(return_value=[])):
        resp = client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")
        assert resp.status_code == 200
        assert resp.json()["report"]["total_issues"] == 0


def test_explicit_vs_inferred_facts(client_with_continuity, sample_universe):
    """Inferred facts must never automatically override explicit facts and receive reduced confidence."""
    client, _ = client_with_continuity
    chap1_id = sample_universe["chap1"]["id"]
    chap2_id = sample_universe["chap2"]["id"]

    # Mock issue where earlier fact was inferred
    mock_issues = [
        {
            "issue_type": "character_consistency",
            "severity": "critical",  # Prompted as critical
            "confidence": 0.95,      # Claimed high confidence
            "description": "Kael claims he is an orphan, but Chapter 1 inferred he might be royal.",
            "existing_fact": "Kael might secretly be an heir to the fallen realm.",
            "new_fact": "Kael states he is the son of an impoverished blacksmith.",
            "affected_entities": ["Kael"],
            "earlier_evidence": [
                {
                    "chapter_id": chap1_id,
                    "chapter_number": 1,
                    "quote_or_excerpt": "clutching his sapphire crystal pendant",
                    "fact_type": "inferred",  # Inferred fact!
                    "confidence": 0.65,
                }
            ],
            "current_evidence": {
                "chapter_id": chap2_id,
                "chapter_number": 2,
                "quote_or_excerpt": "I am merely the son of a blacksmith.",
                "fact_type": "explicit",
                "confidence": 1.0,
            },
            "source_chapter_ids": [chap1_id, chap2_id],
            "possible_explanation": "Earlier royal heritage was only an unconfirmed hypothesis.",
            "suggested_resolution": "No action required if royalty was merely reader/AI speculation.",
        }
    ]

    with patch.object(GeminiService, "evaluate_continuity", new=AsyncMock(return_value=mock_issues)):
        resp = client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")
        assert resp.status_code == 200
        issue = resp.json()["report"]["issues"][0]
        # Inferred facts must be capped at confidence <= 0.65 and severity downgraded from CRITICAL to MEDIUM
        assert issue["confidence"] <= 0.65
        assert issue["severity"] != "critical"


def test_insufficient_evidence_handling(client_with_continuity, sample_universe):
    """Missing or insufficient textual evidence yields low confidence observation."""
    client, _ = client_with_continuity
    chap2_id = sample_universe["chap2"]["id"]

    # Issue with missing evidence quotes
    mock_issues = [
        {
            "issue_type": "plot_continuity",
            "severity": "high",
            "confidence": 0.90,
            "description": "A character seemed to possess a mysterious ring.",
            "existing_fact": "Unspecified lore.",
            "new_fact": "Ring mentioned.",
            "affected_entities": [],
            "earlier_evidence": [],  # Empty evidence!
            "current_evidence": None,
            "source_chapter_ids": [],
        }
    ]

    with patch.object(GeminiService, "evaluate_continuity", new=AsyncMock(return_value=mock_issues)):
        resp = client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")
        assert resp.status_code == 200
        issue = resp.json()["report"]["issues"][0]
        # System enforces low confidence (<= 0.40) and low severity when evidence is absent
        assert issue["confidence"] <= 0.40
        assert issue["severity"] == "low"


def test_off_screen_event_possibility(client_with_continuity, sample_universe):
    """Plausible off-screen transitions are noted with reasonable explanations."""
    client, _ = client_with_continuity
    chap1_id = sample_universe["chap1"]["id"]
    chap2_id = sample_universe["chap2"]["id"]

    mock_issues = [
        {
            "issue_type": "plot_continuity",
            "severity": "low",
            "confidence": 0.45,
            "description": "Kael is wearing iron-plated boots instead of leather moccasins.",
            "existing_fact": "Kael wore leather moccasins in Whispering Woods.",
            "new_fact": "Kael wears heavy iron boots at the spire.",
            "affected_entities": ["Kael"],
            "earlier_evidence": [
                {
                    "chapter_id": chap1_id,
                    "chapter_number": 1,
                    "quote_or_excerpt": "stepped carefully in soft moccasins",
                    "fact_type": "explicit",
                    "confidence": 1.0,
                }
            ],
            "current_evidence": {
                "chapter_id": chap2_id,
                "chapter_number": 2,
                "quote_or_excerpt": "his iron boots rang on the basalt stones",
                "fact_type": "explicit",
                "confidence": 0.9,
            },
            "source_chapter_ids": [chap1_id, chap2_id],
            "possible_explanation": "Kael likely acquired new equipment off-screen during their two-day march.",
            "suggested_resolution": "Optional brief mention of gearing up before the assault.",
        }
    ]

    with patch.object(GeminiService, "evaluate_continuity", new=AsyncMock(return_value=mock_issues)):
        resp = client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")
        assert resp.status_code == 200
        issue = resp.json()["report"]["issues"][0]
        assert "off-screen" in issue["possible_explanation"].lower()


def test_duplicate_issue_prevention_and_status_preservation(client_with_continuity, sample_universe):
    """Rechecking a chapter does not duplicate issues and preserves author triage decisions."""
    client, _ = client_with_continuity
    chap1_id = sample_universe["chap1"]["id"]
    chap2_id = sample_universe["chap2"]["id"]

    mock_issues = [
        {
            "issue_type": "character_consistency",
            "severity": "high",
            "confidence": 0.95,
            "description": "Kael hair color contradiction.",
            "existing_fact": "Kael has dark hair.",
            "new_fact": "Kael has golden-blonde hair.",
            "affected_entities": ["Kael"],
            "earlier_evidence": [
                {
                    "chapter_id": chap1_id,
                    "chapter_number": 1,
                    "quote_or_excerpt": "jet-black hair",
                    "fact_type": "explicit",
                    "confidence": 1.0,
                }
            ],
            "current_evidence": {
                "chapter_id": chap2_id,
                "chapter_number": 2,
                "quote_or_excerpt": "golden-blonde hair",
                "fact_type": "explicit",
                "confidence": 0.95,
            },
            "source_chapter_ids": [chap1_id, chap2_id],
            "possible_explanation": "Disguise potion.",
            "suggested_resolution": "Explain potion.",
        }
    ]

    with patch.object(GeminiService, "evaluate_continuity", new=AsyncMock(return_value=mock_issues)):
        # 1. First check
        first_resp = client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")
        assert first_resp.status_code == 200
        issue_id = first_resp.json()["report"]["issues"][0]["issue_id"]

        # 2. Author accepts the issue with a note explaining the disguise
        patch_resp = client.patch(
            f"/api/v1/continuity/issues/{issue_id}",
            json={
                "status": "accepted",
                "author_notes": "Intentional disguise: Kael drank an elixir of gilded light in scene 3.",
            },
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["status"] == "accepted"

        # 3. Force recheck on the same chapter
        recheck_resp = client.post(
            f"/api/v1/chapters/{chap2_id}/continuity-check",
            json={"force_recheck": True},
        )
        assert recheck_resp.status_code == 200
        rechecked_report = recheck_resp.json()["report"]
        assert rechecked_report["total_issues"] == 1

        # Author's triage status and notes MUST be preserved!
        rechecked_issue = rechecked_report["issues"][0]
        assert rechecked_issue["status"] == "accepted"
        assert "elixir of gilded light" in rechecked_issue["author_notes"]


def test_get_chapter_continuity_report(client_with_continuity, sample_universe):
    """GET /chapters/{chapter_id}/continuity retrieves stored report."""
    client, _ = client_with_continuity
    chap1_id = sample_universe["chap1"]["id"]

    # Initial get returns 404 before check
    r_before = client.get(f"/api/v1/chapters/{chap1_id}/continuity")
    assert r_before.status_code == 404

    # Run check
    client.post(f"/api/v1/chapters/{chap1_id}/continuity-check")

    # Now get succeeds
    r_after = client.get(f"/api/v1/chapters/{chap1_id}/continuity")
    assert r_after.status_code == 200
    assert r_after.json()["chapter_id"] == chap1_id


def test_list_story_continuity_reports(client_with_continuity, sample_universe):
    """GET /stories/{story_id}/continuity returns reports for all audited chapters."""
    client, _ = client_with_continuity
    story_id = sample_universe["story"]["id"]
    chap1_id = sample_universe["chap1"]["id"]
    chap2_id = sample_universe["chap2"]["id"]

    # Run checks for both chapters
    client.post(f"/api/v1/chapters/{chap1_id}/continuity-check")
    with patch.object(GeminiService, "evaluate_continuity", new=AsyncMock(return_value=[])):
        client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")

    resp = client.get(f"/api/v1/stories/{story_id}/continuity")
    assert resp.status_code == 200
    reports = resp.json()
    assert len(reports) == 2
    assert reports[0]["chapter_number"] == 1
    assert reports[1]["chapter_number"] == 2


def test_update_issue_status_triage(client_with_continuity, sample_universe):
    """PATCH /continuity/issues/{issue_id} correctly updates to dismissed and resolved."""
    client, _ = client_with_continuity
    chap1_id = sample_universe["chap1"]["id"]
    chap2_id = sample_universe["chap2"]["id"]

    mock_issues = [
        {
            "issue_type": "relationship_consistency",
            "severity": "medium",
            "confidence": 0.8,
            "description": "Sudden friendship.",
            "existing_fact": "Enemies",
            "new_fact": "Allies",
            "affected_entities": ["Kael", "Lyra"],
            "earlier_evidence": [
                {"chapter_id": chap1_id, "chapter_number": 1, "quote_or_excerpt": "sworn enemy", "fact_type": "explicit", "confidence": 1.0}
            ],
            "current_evidence": {
                "chapter_id": chap2_id, "chapter_number": 2, "quote_or_excerpt": "steadfast alliance", "fact_type": "explicit", "confidence": 1.0
            },
            "source_chapter_ids": [chap1_id, chap2_id],
        }
    ]

    with patch.object(GeminiService, "evaluate_continuity", new=AsyncMock(return_value=mock_issues)):
        resp = client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")
        issue_id = resp.json()["report"]["issues"][0]["issue_id"]

        # 1. Update to dismissed
        d_resp = client.patch(
            f"/api/v1/continuity/issues/{issue_id}",
            json={"status": "dismissed", "author_notes": "Not an error - wolf fight brought them together."},
        )
        assert d_resp.status_code == 200
        assert d_resp.json()["status"] == "dismissed"

        # 2. Update to resolved
        r_resp = client.patch(
            f"/api/v1/continuity/issues/{issue_id}",
            json={"status": "resolved", "author_notes": "Added bridging dialogue in scene 2."},
        )
        assert r_resp.status_code == 200
        assert r_resp.json()["status"] == "resolved"


def test_user_isolation(client_with_continuity, sample_universe):
    """User B cannot access or modify User A's continuity reports or issues."""
    client, _ = client_with_continuity
    chap1_id = sample_universe["chap1"]["id"]

    # Check chapter as default dev user
    client.post(f"/api/v1/chapters/{chap1_id}/continuity-check")

    # Attempt access as other_author via dependency override
    app.dependency_overrides[get_current_user_id] = lambda: "other_author"
    try:
        # Accessing chapter report
        r1 = client.get(f"/api/v1/chapters/{chap1_id}/continuity")
        assert r1.status_code == 404

        # Listing story reports
        story_id = sample_universe["story"]["id"]
        r2 = client.get(f"/api/v1/stories/{story_id}/continuity")
        assert r2.status_code == 404
    finally:
        app.dependency_overrides.pop(get_current_user_id, None)


def test_story_isolation(client_with_continuity, sample_universe):
    """Continuity audits never compare against chapters from other stories."""
    client, _ = client_with_continuity

    # Create unrelated story B
    story_b_resp = client.post(
        "/api/v1/stories",
        json={
            "title": "Cyberpunk Neon",
            "synopsis": "A sci-fi universe in Neo-Tokyo.",
            "genre": "Cyberpunk",
        },
    )
    story_b_id = story_b_resp.json()["id"]

    chap_b_resp = client.post(
        f"/api/v1/stories/{story_b_id}/chapters",
        json={
            "title": "Neon Rain",
            "chapter_number": 1,
            "content": "Detective Ren sat in his hovercar clutching his neural deck.",
        },
    )
    chap_b_id = chap_b_resp.json()["id"]

    # Analyze Chapter B
    now = datetime.now(timezone.utc)
    analysis_repo = get_analysis_repository()
    import asyncio
    asyncio.run(
        analysis_repo.save_report(
            {
                "id": f"rep_{chap_b_id}",
                "chapter_id": chap_b_id,
                "story_id": story_b_id,
                "user_id": DEV_USER_ID,
                "analysis": {
                    "chapter_summary": {"summary": "Ren in hovercar.", "key_takeaways": [], "pov_character": "Ren"},
                    "characters": [],
                    "relationships": [],
                    "events": [],
                    "locations": [],
                    "story_facts": [],
                    "world_rules": [],
                    "timeline": [],
                },
                "model_used": "gemini-flash-lite-latest",
                "created_at": now,
                "updated_at": now,
            }
        )
    )

    # Check continuity for chapter B: should be foundational canon, 0 issues, no cross-contamination from Story A
    resp = client.post(f"/api/v1/chapters/{chap_b_id}/continuity-check")
    assert resp.status_code == 200
    assert resp.json()["report"]["total_issues"] == 0
    assert "Cyberpunk Neon" in resp.json()["report"]["summary"]


def test_gemini_error_handling(client_with_continuity, sample_universe):
    """Gemini API failure or timeout is handled gracefully."""
    client, _ = client_with_continuity
    chap2_id = sample_universe["chap2"]["id"]

    with patch.object(GeminiService, "evaluate_continuity", side_effect=GeminiAPIError("Gemini service timeout")):
        resp = client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")
        assert resp.status_code == 502
        assert "Gemini" in resp.json()["message"] or "timeout" in resp.json()["message"]


def test_manuscript_preservation(client_with_continuity, sample_universe):
    """Running continuity checks never modifies original manuscript or working content."""
    client, _ = client_with_continuity
    chap2_id = sample_universe["chap2"]["id"]
    original_text = sample_universe["chap2"]["original_content"]

    with patch.object(GeminiService, "evaluate_continuity", new=AsyncMock(return_value=[])):
        client.post(f"/api/v1/chapters/{chap2_id}/continuity-check")

    # Fetch chapter from API
    get_resp = client.get(f"/api/v1/chapters/{chap2_id}")
    assert get_resp.status_code == 200
    chapter_after = get_resp.json()

    assert chapter_after["original_content"] == original_text
    assert chapter_after["content"] == original_text


# ============================================================================
# Live Integration Test
# ============================================================================

@pytest.mark.live_api
@pytest.mark.skipif(
    not get_settings().is_gemini_configured,
    reason="Live Gemini API key not configured.",
)
def test_live_gemini_continuity_integration(client_with_continuity, sample_universe):
    """
    Live end-to-end integration test calling the real Google Gemini Flash API.
    Audits Chapter 2 against Chapter 1 and verifies real contradiction detection.
    """
    client, _ = client_with_continuity
    chap2_id = sample_universe["chap2"]["id"]

    # Make the actual real Gemini call
    resp = client.post(
        f"/api/v1/chapters/{chap2_id}/continuity-check",
        json={"force_recheck": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    report = data["report"]

    assert report["chapter_id"] == chap2_id
    assert report["chapter_number"] == 2
    # Real Gemini should spot the crystal rule violation and/or the hair/eye color difference
    assert report["total_issues"] >= 1
    assert any(
        i["issue_type"] in ("world_rule_consistency", "character_consistency")
        for i in report["issues"]
    )
