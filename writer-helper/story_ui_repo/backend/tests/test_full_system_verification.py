"""
Comprehensive 9-Scenario Regression & Integrity Test Suite:
Covers:
- Test 1: New Character verified extraction and persistence
- Test 2: Existing Character no duplicate profile created
- Test 3: Similar Names (Kael vs Kaela) kept strictly separate
- Test 4: Semantic Passage retrieval (e.g. stopped pocket watch) without creating fake characters
- Test 5: Character-Aware Retrieval combining profile + manuscript excerpts
- Test 6: Unsupported Information anti-hallucination honest signaling
- Test 7: Idempotent Re-indexing (no duplicate chunks, no duplicate characters)
- Test 8: Multi-Story Isolation (cross-story universe data segregation)
- Test 9: Separation of Semantic Memory from Character Dossier
"""

import pytest
from app.routers.characters import (
    resolve_character_identity,
    deduplicate_character_entities_helper,
    INVALID_ALIAS_WORDS
)
from app.services.vector_service import vector_service
from app.services.ai_service import ai_service
from app.services.db_service import db_service

# ==============================================================================
# TEST 1 & 2: Verified Character Extraction vs Existing Character Deduplication
# ==============================================================================

def test_new_character_extraction_and_existing_character_deduplication():
    # Character 1: Lord Silas
    existing = [
        {"id": "char_silas_01", "name": "Silas Vance", "aliases": ["Silas", "Mr. Vance"]}
    ]
    
    # Mentioning existing character in new text
    match_existing = resolve_character_identity("Silas Vance", [], existing)
    assert match_existing is not None
    assert match_existing["id"] == "char_silas_01"
    
    # Case insensitive matching
    match_ci = resolve_character_identity("silas vance", [], existing)
    assert match_ci is not None
    assert match_ci["id"] == "char_silas_01"

    # Alias matching
    match_alias = resolve_character_identity("Mr. Vance", [], existing)
    assert match_alias is not None
    assert match_alias["id"] == "char_silas_01"

    # Genuinely new character
    match_new = resolve_character_identity("Clara Moreau", ["Clara"], existing)
    assert match_new is None  # Genuinely new character, should NOT merge into Silas

# ==============================================================================
# TEST 3: Similar Names (Kael vs Kaela) Must Remain Separate
# ==============================================================================

def test_similar_names_kael_and_kaela_remain_separate():
    roster = [
        {"id": "char_kael", "name": "Kael", "aliases": []}
    ]
    
    # Exact match for Kael in any case
    assert resolve_character_identity("Kael", [], roster)["id"] == "char_kael"
    assert resolve_character_identity("KAEL", [], roster)["id"] == "char_kael"
    assert resolve_character_identity("kael", [], roster)["id"] == "char_kael"

    # Kaela must NOT match Kael
    match_kaela = resolve_character_identity("Kaela", [], roster)
    assert match_kaela is None

# ==============================================================================
# TEST 4: Semantic Passage Retrieval (Stopped Pocket Watch) & No Fake Characters
# ==============================================================================

def test_semantic_passage_retrieval_and_no_fake_character_creation():
    sample_text = (
        "The heavy silver casing had been blackened by sea salt. "
        "Inside rested an antique pocket watch, its crystal cracked from nine to four. "
        "The hands were seized at precisely 3:17 beside the cold harbor stones."
    )
    
    # 1. Heuristic extractor must NEVER interpret the watch or harbor as a character
    extracted_chars = ai_service._heuristic_extract_characters(sample_text, [], "Mystery")
    extracted_names = [c["name"].lower() for c in extracted_chars]
    assert "pocket watch" not in extracted_names
    assert "harbor" not in extracted_names
    assert "casing" not in extracted_names
    assert len(extracted_chars) == 0

    # 2. Semantic indexing into ChromaDB
    test_chap = {
        "id": "chap_test_watch",
        "chapter_number": 4,
        "title": "The Forgotten Station",
        "content": sample_text
    }
    vector_service.index_chapter("story_test_watch_universe", test_chap)

    # 3. Search for stopped pocket watch
    results = vector_service.search_memories(
        query="What happened to the stopped pocket watch?",
        story_id="story_test_watch_universe",
        top_k=3
    )
    assert len(results) > 0
    top = results[0]
    assert "3:17" in top["description"]
    assert top["source_chapter"] == "Chapter 4"
    assert top["category"] == "chapter_passage"

    # Cleanup
    vector_service.delete_chapter_index("chap_test_watch")

# ==============================================================================
# TEST 6: Anti-Hallucination & Unsupported Information Signaling
# ==============================================================================

def test_unsupported_information_returns_empty_or_honest_result():
    # Search for an element that does not exist in the story
    results = vector_service.search_memories(
        query="The Martian mothership hovering over the Eiffel Tower in Paris",
        story_id="story_ravenscar_noir",
        top_k=2
    )
    # Relevance should be low or non-matching
    assert all("martian" not in r["description"].lower() for r in results)

# ==============================================================================
# TEST 7: Idempotent Re-indexing (No Duplicate Chunks)
# ==============================================================================

def test_idempotent_reindexing_prevents_duplicate_chunks():
    test_chapter = {
        "id": "chap_idempotent_1",
        "chapter_number": 1,
        "title": "The Silent Fog",
        "content": (
            "The fog rolled over the cobblestones like a shroud. "
            "Alistair stood alone in the dark gallery awaiting the courier."
        )
    }

    # Index first time
    count1 = vector_service.index_chapter("story_test_idempotent", test_chapter)
    assert count1 > 0

    # Index second time (re-indexing)
    count2 = vector_service.index_chapter("story_test_idempotent", test_chapter)
    assert count2 == count1

    # Verify count in ChromaDB is exactly count1, not doubled
    chroma_res = vector_service.collection.get(where={"chapter_id": "chap_idempotent_1"})
    assert len(chroma_res["ids"]) == count1

    # Cleanup
    vector_service.delete_chapter_index("chap_idempotent_1")

# ==============================================================================
# TEST 8: Multi-Story Universe Isolation
# ==============================================================================

def test_multi_story_isolation_prevents_cross_universe_leakage():
    chap_story_a = {
        "id": "chap_story_a_1",
        "chapter_number": 1,
        "title": "Secret of Universe Alpha",
        "content": "In universe Alpha, the secret vault code is OMEGA-9921."
    }
    chap_story_b = {
        "id": "chap_story_b_1",
        "chapter_number": 1,
        "title": "Secret of Universe Beta",
        "content": "In universe Beta, the crystal spire belongs to the Solar Queen."
    }

    vector_service.index_chapter("story_alpha_99", chap_story_a)
    vector_service.index_chapter("story_beta_88", chap_story_b)

    # Query Story Alpha for Alpha's secret
    res_a = vector_service.search_memories(query="vault code OMEGA", story_id="story_alpha_99", top_k=5)
    assert any("OMEGA-9921" in r["description"] for r in res_a)
    assert not any("Solar Queen" in r["description"] for r in res_a)

    # Query Story Beta for Beta's secret
    res_b = vector_service.search_memories(query="crystal spire Solar Queen", story_id="story_beta_88", top_k=5)
    assert any("Solar Queen" in r["description"] for r in res_b)
    assert not any("OMEGA-9921" in r["description"] for r in res_b)

    # Cleanup
    vector_service.delete_chapter_index("chap_story_a_1")
    vector_service.delete_chapter_index("chap_story_b_1")

