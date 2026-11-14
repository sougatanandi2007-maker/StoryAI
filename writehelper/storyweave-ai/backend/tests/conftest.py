"""
Shared test configuration and fixtures for StoryWeave AI test suite.
"""

import os
# Ensure tests run with offline Hugging Face cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest
from unittest.mock import MagicMock
import app.services.chapter_service as cs

# Fast deterministic mock for DistilBART in unit tests
@pytest.fixture(autouse=True)
def mock_summarization_for_tests(monkeypatch):
    """Ensure tests run instantly without downloading or initializing heavy local DistilBART weights."""
    mock_service = MagicMock()
    mock_service.model_name = "sshleifer/distilbart-cnn-12-6"
    mock_service.summarize_chapter.return_value = "Mock chapter summary capturing key events and character interactions."
    monkeypatch.setattr(cs, "get_summarization_service", lambda: mock_service)
    yield mock_service


from app.database.repositories.dependencies import (
    get_story_repository,
    get_chapter_repository,
    get_analysis_repository,
    get_continuity_repository,
    get_character_repository,
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


@pytest.fixture(autouse=True)
def reset_all_repositories():
    """Clear all in-memory repository storage before each test."""
    get_story_repository().clear()
    get_chapter_repository().clear()
    get_analysis_repository().clear()
    get_continuity_repository().clear()
    get_character_repository().clear()
    yield


@pytest.fixture
def sample_analysis_result() -> ChapterAnalysisResult:
    """Provide a valid, comprehensive ChapterAnalysisResult for testing."""
    return ChapterAnalysisResult(
        chapter_summary=ChapterSummary(
            summary=(
                "Kael arrives at the Whispering Woods seeking the Celestial Crystal of Eldoria. "
                "He meets Lyra, a guardian archer, who warns him of the Blood Eclipse. "
                "Together they fend off shadow beasts and form a wary alliance."
            ),
            key_takeaways=[
                "Kael enters the Whispering Woods seeking the crystal",
                "Lyra warns him about the impending Blood Eclipse",
                "Shadow beasts attack, confirming the forest is corrupted",
            ],
            pov_character="Kael",
            pacing_assessment="Starts with tense exploration, escalates to rapid combat.",
        ),
        characters=[
            ExtractedCharacter(
                name="Kael of Riverdell",
                aliases=["The Crystal Seeker", "Kael"],
                description="Young man with dark hair, weathered leather armor, carrying a sapphire pendant.",
                personality_traits=["Determined", "Cautious", "Resourceful"],
                motivations=["Save his village from the creeping blight"],
                goals=["Retrieve the Celestial Crystal from the forest ruins"],
                abilities=["Tracking", "Ether-sensing"],
                evidence="Kael clutching his crystal dagger, stepped carefully over the frozen roots.",
            ),
            ExtractedCharacter(
                name="Lyra",
                aliases=["Guardian of the Glade"],
                description="Tall elven archer with silver-threaded cloak and bow made of weirwood.",
                personality_traits=["Vigilant", "Distrusting", "Loyal"],
                motivations=["Protect the sanctuary from defilement"],
                goals=["Prevent mortals from touching the sacred altar"],
                abilities=["Archery", "Camouflage"],
                evidence="From the canopy above, an arrow struck the dirt inches from Kael's boot. 'Turn back, mortal,' Lyra warned.",
            ),
        ],
        relationships=[
            ExtractedRelationship(
                character_1="Kael of Riverdell",
                character_2="Lyra",
                relationship_type="ally",
                description="Initial hostility gives way to reluctant cooperation against shadow beasts.",
                evidence="'If you wish to live through the eclipse,' Lyra muttered, 'we fight together.'",
            ),
        ],
        events=[
            ExtractedEvent(
                title="Ambush in the Whispering Pines",
                description="Shadow wolves emerge from the mist and ambush Kael and Lyra.",
                participants=["Kael of Riverdell", "Lyra"],
                location="Whispering Woods",
                chronological_order=1,
                significance="Forces Kael and Lyra to unite despite their mutual suspicion.",
                evidence="Three shadow beasts sprang from the darkness, their crimson eyes flashing.",
            ),
        ],
        locations=[
            ExtractedLocation(
                name="Whispering Woods",
                description="An ancient forest shrouded in perpetual mist and frost-bitten pine needles.",
                significance="Site of the hidden celestial ruins.",
                evidence="The twisted pines of Whispering Woods seemed to murmur in an forgotten tongue.",
            ),
        ],
        story_facts=[
            ExtractedFact(
                fact="The Blood Eclipse occurs once every century and weakens protective wards.",
                category="lore",
                fact_type=FactType.EXPLICIT,
                confidence=1.0,
                evidence="When the moon turns red, the ancient wards crumble like ash.",
            ),
            ExtractedFact(
                fact="Kael may possess royal blood due to his natural ether-sensing ability.",
                category="secret",
                fact_type=FactType.INFERRED,
                confidence=0.75,
                evidence="Lyra stared at the glowing sapphire pendant on Kael's chest: 'Only the lost line of Riverdell can awaken that stone.'",
            ),
        ],
        world_rules=[
            ExtractedWorldRule(
                rule="Magic requires physical crystal catalysts; mortals cannot cast raw ether without a focus.",
                category="magic",
                evidence="No spell could be woven without channeling the energy through refined celestial glass.",
            ),
        ],
        timeline=[
            ExtractedTimelineEvent(
                relative_time="At dusk, on the eve of the Blood Eclipse",
                is_flashback=False,
                uncertain_dates=False,
                time_anchor="Blood Eclipse",
                evidence="As the twilight sun dipped below the mountain ridge, the red tinge began to seep into the clouds.",
            ),
        ],
    )
