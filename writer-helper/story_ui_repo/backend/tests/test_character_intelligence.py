"""
Test Suite: StoryWeave AI Intelligent Character Intelligence & Dossier Resolution
Covers:
1. Exact Canonical Name Matching & Deduplication
2. Alias & Nickname Resolution ("Julian Calder" == "Mr. Calder" == "Calder" == "Julian")
3. Multi-word nickname resolution ("Lullu" == "Lullu Dickwala")
4. Disambiguation & Separation of distinct people with shared surnames (e.g., "Julian Vance" vs "Silas Vance")
5. Status History Timeline transitions (e.g. Presumed dead -> Alive -> Deceased)
6. Prevention of self-connections & reciprocal duplicate connections
7. Idempotent synchronization across chapters
"""

import pytest
from app.routers.characters import (
    resolve_character_identity,
    record_character_status_transition,
    normalize_name_tokens,
)

def test_token_normalization():
    tokens = normalize_name_tokens("Mr. Julian Calder")
    assert tokens == ["julian", "calder"]

    tokens_dr = normalize_name_tokens("Dr. Silas Vance, Esq.")
    assert "silas" in tokens_dr and "vance" in tokens_dr

def test_exact_canonical_name_match():
    existing = [
        {"id": "char_1", "name": "Julian Calder", "aliases": []}
    ]
    matched = resolve_character_identity("Julian Calder", [], existing)
    assert matched is not None
    assert matched["id"] == "char_1"

def test_case_insensitive_name_match():
    existing = [
        {"id": "char_1", "name": "Julian Calder", "aliases": []}
    ]
    matched = resolve_character_identity("julian calder", [], existing)
    assert matched is not None
    assert matched["id"] == "char_1"

def test_alias_match_existing_character():
    existing = [
        {"id": "char_1", "name": "Julian Calder", "aliases": ["The Architect", "Calder"]}
    ]
    matched = resolve_character_identity("The Architect", [], existing)
    assert matched is not None
    assert matched["id"] == "char_1"

def test_candidate_aliases_match_existing():
    existing = [
        {"id": "char_1", "name": "Julian Calder", "aliases": []}
    ]
    matched = resolve_character_identity("Calder", ["Julian Calder"], existing)
    assert matched is not None
    assert matched["id"] == "char_1"

def test_honorific_mr_calder_resolves_to_julian_calder():
    existing = [
        {"id": "char_1", "name": "Julian Calder", "aliases": []}
    ]
    matched = resolve_character_identity("Mr. Calder", [], existing)
    assert matched is not None
    assert matched["id"] == "char_1"

def test_single_given_name_resolves_if_unique():
    existing = [
        {"id": "char_1", "name": "Julian Calder", "aliases": []}
    ]
    matched = resolve_character_identity("Julian", [], existing)
    assert matched is not None
    assert matched["id"] == "char_1"

def test_single_surname_resolves_if_unique():
    existing = [
        {"id": "char_1", "name": "Julian Calder", "aliases": []}
    ]
    matched = resolve_character_identity("Calder", [], existing)
    assert matched is not None
    assert matched["id"] == "char_1"

def test_lullu_resolves_to_lullu_dickwala():
    existing = [
        {"id": "char_2", "name": "Lullu Dickwala", "aliases": ["Lullu"]}
    ]
    matched = resolve_character_identity("Lullu", [], existing)
    assert matched is not None
    assert matched["id"] == "char_2"

def test_distinct_people_with_same_surname_never_merged():
    """
    Julian Vance and Silas Vance share the surname 'Vance' but are distinct characters.
    Neither should falsely merge into the other when a new reference appears.
    """
    existing = [
        {"id": "char_vance_1", "name": "Julian Vance", "aliases": []},
        {"id": "char_vance_2", "name": "Silas Vance", "aliases": []}
    ]
    # New mention of Silas Vance should match Silas Vance, not Julian
    matched_silas = resolve_character_identity("Silas Vance", [], existing)
    assert matched_silas is not None
    assert matched_silas["id"] == "char_vance_2"

    # Ambiguous surname 'Vance' when both exist should NOT blindly merge into one
    matched_ambiguous = resolve_character_identity("Mr. Vance", [], existing)
    assert matched_ambiguous is None  # Safe: flag or leave unmerged rather than corrupted

def test_distinct_similar_names_never_merged():
    existing = [
        {"id": "char_k1", "name": "Kael", "aliases": []}
    ]
    matched = resolve_character_identity("Kaela", [], existing)
    assert matched is None

def test_status_history_transition_preserves_provenance():
    char = {
        "id": "char_1",
        "name": "Marcus Kane",
        "current_status": "Alive",
        "status_history": [
            {"status": "Alive", "chapter": 1, "evidence": "Seen at station", "type": "explicit"}
        ]
    }
    # Chapter 3: Presumed dead
    changed = record_character_status_transition(char, "Presumed dead", chapter_num=3, evidence_text="Body lost in river")
    assert changed is True
    assert char["current_status"] == "Presumed dead"
    assert len(char["status_history"]) == 2
    assert char["status_history"][-1]["chapter"] == 3

    # Chapter 5: Revealed alive
    changed2 = record_character_status_transition(char, "Alive", chapter_num=5, evidence_text="Seen in safehouse")
    assert changed2 is True
    assert char["current_status"] == "Alive"
    assert len(char["status_history"]) == 3

    # Chapter 7: Deceased
    changed3 = record_character_status_transition(char, "Deceased", chapter_num=7, evidence_text="Mortally wounded")
    assert changed3 is True
    assert char["current_status"] == "Deceased"
    assert len(char["status_history"]) == 4

def test_status_history_idempotent():
    char = {
        "id": "char_1",
        "name": "Marcus Kane",
        "current_status": "Alive",
        "status_history": [
            {"status": "Alive", "chapter": 1, "evidence": "Seen at station", "type": "explicit"}
        ]
    }
    # Calling with same status again should not duplicate history entries
    changed = record_character_status_transition(char, "Alive", chapter_num=1)
    assert changed is False
    assert len(char["status_history"]) == 1

def test_relationship_directionality_and_deduplication():
    from app.services.ai_service import AIService
    service = AIService()

    # Characters in story
    characters = [
        {"id": "c_julian", "name": "Julian Calder"},
        {"id": "c_saptarshi", "name": "Saptarshi"},
        {"id": "c_soumik", "name": "soumik"},
        {"id": "c_sagnik", "name": "Sagnik"},
        {"id": "c_mashir", "name": "Mashir Chele"},
        {"id": "c_lullu", "name": "Lullu Dickwala"}
    ]
    chapters = [
        {
            "chapter_number": 1,
            "title": "Howrah Terminal",
            "content": "Julian Calder loved Saptarshi like a brother. Sagnik clung affectionately to Saptarshi declaring his eternal devotion. Suddenly, Soumik aimed his sniper rifle from the catwalk and shot both Saptarshi and Sagnik dead. Mashir Chele nodded to Soumik as they planned the ambush together."
        }
    ]

    connections = service.analyze_story_relationships_and_connections(characters, chapters, {"title": "Howrah 2150"})
    assert len(connections) >= 4

    # Verify directional killer/victim edges
    killer_edges = [c for c in connections if c.get("type") == "killer_victim"]
    assert len(killer_edges) >= 1
    for k in killer_edges:
        assert k.get("direction") == "directional"
        assert k.get("sourceId") in ["c_soumik", "c_mashir"]

    # Verify friends / romance edges
    romance_edges = [c for c in connections if c.get("type") == "romance"]
    assert len(romance_edges) >= 1
    for r in romance_edges:
        assert r.get("sourceId") in ["c_saptarshi", "c_sagnik"]
        assert r.get("targetId") in ["c_saptarshi", "c_sagnik"]

def test_self_connection_and_invalid_ids_rejection():
    """Ensure that cards never connect to themselves and no dummy IDs can be linked."""
    from app.routers.characters import deduplicate_character_entities_helper
    
    raw_characters = [
        {"id": "c1", "name": "Julian Calder", "aliases": ["Julian", "Calder"]},
        {"id": "c2", "name": "Julian", "aliases": []},
        {"id": "c3", "name": "Saptarshi", "aliases": []}
    ]
    board = {
        "positions": {"c1": {"x": 10, "y": 10}, "c2": {"x": 20, "y": 20}, "c3": {"x": 30, "y": 30}},
        "connections": [
            {"id": "conn1", "sourceId": "c1", "targetId": "c2", "label": "Self Duplicate"}, # Should be removed as self-link after merge
            {"id": "conn2", "sourceId": "c1", "targetId": "c3", "label": "Friends"},
            {"id": "conn3", "sourceId": "c3", "targetId": "c3", "label": "Self loop"}, # Self connection
            {"id": "conn4", "sourceId": "c1", "targetId": "nonexistent_id", "label": "Ghost"}
        ]
    }

    canonical_chars, purged, updated_board = deduplicate_character_entities_helper(raw_characters, board)
    assert len(canonical_chars) == 2
    assert "c2" in purged

    valid_conns = updated_board.get("connections", [])
    # Self-links (c1->c1) and non-existent IDs must be completely purged
    for c in valid_conns:
        assert c["sourceId"] != c["targetId"]
        assert c["targetId"] in [ch["id"] for ch in canonical_chars]
        assert c["sourceId"] in [ch["id"] for ch in canonical_chars]


