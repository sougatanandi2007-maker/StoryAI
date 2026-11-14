"""
Comprehensive Test Suite for:
1. Problem A: Lossless Character Consolidation & False-Merge Prevention
2. Problem B: Universe Memory Vector Search & Evidence-Grounded Q&A
"""

import pytest
from app.routers.characters import (
    resolve_character_identity,
    deduplicate_character_entities_helper,
    INVALID_ALIAS_WORDS
)
from app.services.vector_service import vector_service

# ==============================================================================
# PART A: Character Lossless Consolidation & Safeguards
# ==============================================================================

def test_invalid_alias_words_not_merged():
    """Ensure generic pronouns, colloquial honorifics ('bhai', 'tu') do not trigger false identity merges."""
    assert "bhai" in INVALID_ALIAS_WORDS
    assert "tu" in INVALID_ALIAS_WORDS
    assert "he" in INVALID_ALIAS_WORDS
    assert "she" in INVALID_ALIAS_WORDS

    existing_saptarshi = {
        "id": "char_saptarshi",
        "name": "Saptarshi",
        "aliases": ["Saptarshi", "bhai", "tu"]  # Even if malformed aliases existed in store
    }
    candidates = [existing_saptarshi]

    # Distinct character Sougat with colloquial greeting "Sougat bhai"
    match = resolve_character_identity("Sougat", ["Sougat bhai", "bhai", "tu"], candidates)
    # Sougat must NOT match Saptarshi
    assert match is None

def test_token_overlap_safeguard_prevents_distinct_name_merges():
    """Ensure two distinct people with different names aren't merged just because of dirty alias lists."""
    char_a = {
        "id": "char_alok",
        "name": "Alok Kumar",
        "aliases": ["Alok", "bhai"]
    }
    # Sougat is totally distinct from Alok
    match = resolve_character_identity("Sougat", ["bhai"], [char_a])
    assert match is None

def test_lossless_character_consolidation():
    """Verify that merging duplicates is 100% lossless: no fields, traits, or status history lost."""
    primary = {
        "id": "char_julian_1",
        "name": "Julian Calder",
        "aliases": ["Julian"],
        "physical_description": "Tall man with grey coat.",
        "personality_traits": ["Methodical", "Stoic"],
        "backstory": "Former imperial surveyor.",
        "motivations": "Uncover the conspiracy.",
        "goals": "Reach the terminal.",
        "fears": "Public exposure.",
        "secrets": "Possesses a private telegraph key.",
        "status": "Alive",
        "first_appearance": "Chapter 1",
        "last_appearance": "Chapter 1",
        "appearances": ["Chapter 1"],
        "status_history": [
            {"chapter": "Chapter 1", "from_status": "Unknown", "to_status": "Alive", "reason": "Arrives at docks"}
        ],
        "relationships": [
            {"target_name": "Silas Vance", "type": "associate"}
        ]
    }

    duplicate = {
        "id": "char_julian_2",
        "name": "Mr. Calder",
        "aliases": ["Calder", "The Surveyor"],
        "physical_description": "Scar over left eye; brass spectacles.",
        "personality_traits": ["Sharp-eyed", "Stoic"],
        "backstory": "Exiled from ministry after the rail collapse.",
        "motivations": "Clear his family name.",
        "goals": "Secure the ledger.",
        "fears": "Betrayal.",
        "secrets": "Knows cipher 44.",
        "status": "Injured",
        "first_appearance": "Chapter 2",
        "last_appearance": "Chapter 3",
        "appearances": ["Chapter 2", "Chapter 3"],
        "status_history": [
            {"chapter": "Chapter 3", "from_status": "Alive", "to_status": "Injured", "reason": "Shrapnel blast"}
        ],
        "relationships": [
            {"target_name": "Elena", "type": "informant"}
        ]
    }

    board_data = {
        "positions": {
            "char_julian_1": {"x": 100, "y": 100},
            "char_julian_2": {"x": 200, "y": 200},
            "char_elena": {"x": 300, "y": 300}
        },
        "connections": [
            {"sourceId": "char_julian_2", "targetId": "char_elena", "type": "informant"}
        ]
    }

    elena = {
        "id": "char_elena",
        "name": "Elena",
        "aliases": ["Elena"],
        "status": "Alive"
    }

    canonical_list, purged_ids, updated_board = deduplicate_character_entities_helper(
        [primary, duplicate, elena],
        board_data
    )

    # 1. Duplicate purged, canonical preserved
    assert len(canonical_list) == 2
    assert "char_julian_2" in purged_ids
    merged = canonical_list[0]
    assert merged["id"] == "char_julian_1"

    # 2. Aliases merged losslessly
    assert "The Surveyor" in merged["aliases"]
    assert "Calder" in merged["aliases"]

    # 3. Traits & descriptions merged
    assert "Methodical" in merged["personality_traits"]
    assert "Sharp-eyed" in merged["personality_traits"]
    assert "Scar over left eye" in merged["physical_description"]
    assert "Former imperial surveyor" in merged["backstory"]
    assert "Exiled from ministry" in merged["backstory"]

    # 4. Motivations, goals, fears, secrets combined
    assert "Uncover the conspiracy" in merged["motivations"]
    assert "Clear his family name" in merged["motivations"]
    assert "Possesses a private telegraph key" in merged["secrets"]
    assert "Knows cipher 44" in merged["secrets"]

    # 5. Appearances span
    assert "Chapter 1" in merged["appearances"]
    assert "Chapter 2" in merged["appearances"]
    assert "Chapter 3" in merged["appearances"]

    # 6. Status history merged
    assert len(merged["status_history"]) == 2
    assert merged["status"] == "Injured"

    # 7. Relationships merged
    rel_targets = [r["target_name"] for r in merged["relationships"]]
    assert "Silas Vance" in rel_targets
    assert "Elena" in rel_targets

    # 8. Board connections remapped
    assert updated_board["connections"][0]["sourceId"] == "char_julian_1"


# ==============================================================================
# PART B: Universe Memory Vector Search & Grounding Tests
# ==============================================================================

def test_chapter_chunking_and_dense_indexing():
    """Verify that chapter indexing splits into semantic chunks with metadata."""
    sample_chapter = {
        "id": "chap_test_101",
        "chapter_number": 5,
        "title": "The Glass Citadel",
        "content": (
            "The ancient vault under the breakwater was sealed with cold mercury. "
            "Julian Calder touched the brass dial on the bulkhead door. "
            "Alistair stood vigil behind him, clutching the iron lantern. "
            "Inside the chamber lay the lost navigational logs of the Nemesis."
        )
    }

    # Index into ChromaDB
    indexed = vector_service.index_chapter("story_test_suite", sample_chapter)
    assert indexed > 0

    # Search for semantic concept with vocabulary mismatch ("maritime journal" vs "navigational logs")
    results = vector_service.search_memories("story_test_suite", "maritime records under water breakwater", top_k=3)
    assert len(results) > 0
    top = results[0]
    assert "The Glass Citadel" in top.get("title", "") or "mercury" in top.get("description", "")
    assert top.get("source_chapter") == "Chapter 5"

    # Cleanup test index
    deleted = vector_service.delete_chapter_index("chap_test_101")
    assert deleted > 0

