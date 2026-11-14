from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from ..models import Character, CharacterCreate, CharacterUpdate
from ..services.db_service import db_service
from ..services.ai_service import ai_service
import uuid

router = APIRouter(tags=["Characters"])

@router.get("/api/stories/{story_id}/characters", response_model=List[Character])
def get_story_characters(story_id: str):
    return db_service.get_characters(story_id)

import re
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from ..models import Character, CharacterCreate, CharacterUpdate
from ..services.db_service import db_service
from ..services.ai_service import ai_service

router = APIRouter(tags=["Characters"])

TITLES_HONORIFICS = {
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss", "dr", "dr.", 
    "doctor", "inspector", "detective", "captain", "lord", "lady",
    "sir", "madam", "madame", "master", "officer", "agent", "professor", "prof", "prof."
}

# Generic pronouns and single filler words must NEVER act as identity-matching aliases
INVALID_ALIAS_WORDS = {
    "bhai", "tu", "tujh", "tera", "meri", "hum", "bro", "dude", "man", "guy",
    "he", "she", "him", "her", "they", "them", "you", "it"
}

def normalize_name_tokens(name: str) -> List[str]:
    """Tokenize name removing punctuation and filtering out titles."""
    cleaned = re.sub(r"[^\w\s]", "", name.lower())
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    return [t for t in tokens if t not in TITLES_HONORIFICS]

def resolve_character_identity(
    candidate_name: str,
    candidate_aliases: List[str],
    existing_characters: List[Dict[str, Any]],
    context_text: str = ""
) -> Optional[Dict[str, Any]]:
    """
    Intelligent Entity Resolution System:
    1. Exact canonical name match (case-insensitive).
    2. Exact alias match (matches candidate name against existing aliases or candidate aliases against existing names/aliases).
    3. Title-stripped exact match (e.g. 'Mr. Calder' -> 'Calder', matches 'Julian Calder' if Calder is unique).
    4. Surname / single-name match ONLY when uniquely identifying a single existing character with that surname in the story.
    5. Never merge two distinct characters who share a surname but have different first names (e.g. Julian Calder vs Marcus Calder, or Silas Vance vs Julian Vance).
    6. Never merge distinct characters simply because names are similar.
    """
    c_clean = candidate_name.strip()
    c_lower = c_clean.lower()
    if not c_lower:
        return None

    all_cand_names = [c_lower] + [a.strip().lower() for a in candidate_aliases if a.strip()]

    # Generic pronouns and single filler words must NEVER act as identity-matching aliases
    INVALID_ALIAS_WORDS = {"bhai", "tu", "tujh", "tera", "meri", "hum", "bro", "dude", "man", "guy", "he", "she", "him", "her", "they", "them", "you", "it"}

    # Stage 1: Exact Name or Known Alias Match
    for ex in existing_characters:
        ex_name = ex.get("name", "").strip().lower()
        ex_aliases = [a.strip().lower() for a in ex.get("aliases", []) if a.strip() and a.strip().lower() not in INVALID_ALIAS_WORDS]
        if ex.get("alias") and ex.get("alias").strip().lower() not in INVALID_ALIAS_WORDS:
            ex_aliases.append(ex.get("alias", "").strip().lower())

        # Check if candidate name matches ex_name
        if c_lower == ex_name:
            return ex

        # Check if candidate name matches ex_aliases ONLY IF it's not a common invalid word and doesn't conflict with ex_name
        if c_lower in ex_aliases and c_lower not in INVALID_ALIAS_WORDS:
            # If the candidate name is a distinct multi-letter capitalized word completely different from ex_name,
            # ensure it doesn't match an independent character's primary name or token
            cand_tokens = normalize_name_tokens(c_clean)
            ex_tokens = normalize_name_tokens(ex.get("name", ""))
            # If candidate tokens have zero overlap with ex_name tokens, only match if candidate alias is explicit title/alias (e.g. 'The Architect')
            if any(t in ex_tokens for t in cand_tokens) or len(cand_tokens) > 1 or c_lower == ex.get("alias", "").lower():
                return ex

        # Check if candidate aliases match ex_name or valid ex_aliases
        for cand_a in all_cand_names:
            if cand_a in INVALID_ALIAS_WORDS:
                continue
            if cand_a == ex_name:
                return ex
            if cand_a in ex_aliases:
                cand_a_tokens = normalize_name_tokens(cand_a)
                ex_tokens = normalize_name_tokens(ex.get("name", ""))
                if any(t in ex_tokens for t in cand_a_tokens) or len(cand_a_tokens) > 1:
                    return ex

    # Stage 2: Token-based match with Title Stripping
    cand_tokens = normalize_name_tokens(c_clean)
    if not cand_tokens:
        return None

    # Check against each existing character's normalized tokens
    matches = []
    for ex in existing_characters:
        ex_name = ex.get("name", "").strip()
        ex_tokens = normalize_name_tokens(ex_name)
        if not ex_tokens:
            continue

        # If exact token sequence match
        if cand_tokens == ex_tokens:
            matches.append(ex)
            continue

        # If candidate is a single surname or given name (e.g. "Calder" or "Julian")
        # Ensure single name is not an edit-distance variant of a distinct character (e.g. Kael vs Kaela)
        if len(cand_tokens) == 1 and cand_tokens[0] in ex_tokens:
            matches.append(ex)
        # Or if existing is a single name and candidate is full name (e.g. existing "Julian", candidate "Julian Calder")
        elif len(ex_tokens) == 1 and ex_tokens[0] in cand_tokens:
            matches.append(ex)

    # Disallow false positive matches where names differ by letters (e.g. Kael != Kaela)
    if len(matches) == 1:
        matched_char = matches[0]
        ex_clean = matched_char.get("name", "").strip().lower()
        # If candidate name and matched name have same token count > 0 but different strings (e.g. Kael vs Kaela)
        if len(cand_tokens) == 1 and len(normalize_name_tokens(ex_clean)) == 1:
            if cand_tokens[0] != normalize_name_tokens(ex_clean)[0]:
                return None

    # If exactly ONE unique character matches, and there is NO conflicting distinct character
    if len(matches) == 1:
        matched_char = matches[0]
        # Verify there isn't another character sharing this surname with a different first name
        m_tokens = normalize_name_tokens(matched_char.get("name", ""))
        shared_surname = m_tokens[-1] if len(m_tokens) > 1 else (m_tokens[0] if m_tokens else "")
        if shared_surname and len(cand_tokens) == 1 and cand_tokens[0] == shared_surname:
            # Check how many characters share this surname
            same_surname_chars = [
                c for c in existing_characters 
                if normalize_name_tokens(c.get("name", "")) and normalize_name_tokens(c.get("name", ""))[-1] == shared_surname
            ]
            if len(same_surname_chars) > 1:
                # Ambiguous! Do NOT merge automatically, keep separate.
                return None
        return matched_char

    return None

VALID_NARRATIVE_STATUSES = {
    "Alive", "Presumed dead", "Deceased", "Missing", "Unknown", 
    "Injured", "Resurrected / returned", "Unconfirmed"
}

def record_character_status_transition(
    char: Dict[str, Any],
    new_status: str,
    chapter_num: int,
    evidence_text: str = "",
    status_type: str = "explicit"
) -> bool:
    """
    Records a state transition in character's status_history if changed,
    preserving chronological narrative timeline provenance.
    """
    clean_status = new_status.strip().capitalize() if new_status else "Alive"
    # Map synonyms to canon narrative statuses
    lower_s = clean_status.lower()
    if "presumed" in lower_s or "thought dead" in lower_s:
        clean_status = "Presumed dead"
    elif "dead" in lower_s or "deceased" in lower_s or "killed" in lower_s:
        clean_status = "Deceased"
    elif "missing" in lower_s or "lost" in lower_s:
        clean_status = "Missing"
    elif "injured" in lower_s or "wounded" in lower_s:
        clean_status = "Injured"
    elif "resurrect" in lower_s or "returned" in lower_s:
        clean_status = "Resurrected / returned"
    elif "unconfirmed" in lower_s:
        clean_status = "Unconfirmed"
    elif "alive" in lower_s:
        clean_status = "Alive"
    elif clean_status not in VALID_NARRATIVE_STATUSES:
        clean_status = "Unknown"

    history = char.get("status_history", [])
    current_status = char.get("current_status", "Alive")

    if current_status != clean_status or not history:
        history.append({
            "status": clean_status,
            "chapter": chapter_num,
            "evidence": evidence_text[:300] if evidence_text else f"Observed in Chapter {chapter_num}",
            "type": status_type
        })
        char["status_history"] = history
        char["current_status"] = clean_status
        return True
    return False

@router.get("/api/stories/{story_id}/characters", response_model=List[Character])
def get_story_characters(story_id: str):
    return db_service.get_characters(story_id)

@router.post("/api/stories/{story_id}/characters/sync")
def sync_characters_from_manuscript(story_id: str):
    """
    Automated Chapter Character Extraction & Synchronization:
    Scans all chapters of the story to extract real characters mentioned in the manuscript,
    deduplicating using canonical entity resolution, updating status histories,
    and never inventing characters or merging distinct individuals.
    """
    story = db_service.get_story(story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
        
    genre = story.get("genre", "Fiction")
    chapters = db_service.get_chapters(story_id)
    existing_chars = db_service.get_characters(story_id)
    
    newly_added = []
    updated_chars = []
    
    # Sort chapters in chronological order
    sorted_chapters = sorted(chapters, key=lambda c: c.get("chapter_number", 0))

    for chap in sorted_chapters:
        c_num = chap.get("chapter_number", 1)
        content = chap.get("content", "").strip()
        if len(content) < 30:
            continue
            
        all_known = existing_chars + newly_added
        extracted = ai_service.extract_characters_from_text(
            text=content,
            existing_characters=all_known,
            story_genre=genre
        )
        
        for item in extracted:
            c_name = item.get("name", "").strip()
            if not c_name:
                continue

            c_aliases = item.get("aliases", [])
            if item.get("alias") and item.get("alias") not in c_aliases:
                c_aliases.append(item.get("alias"))

            matched = resolve_character_identity(
                candidate_name=c_name,
                candidate_aliases=c_aliases,
                existing_characters=all_known,
                context_text=content
            )

            if matched:
                # Merge aliases
                existing_aliases = matched.get("aliases", [])
                for a in c_aliases:
                    if a.strip() and a.strip().lower() != matched.get("name", "").lower():
                        if a.strip().lower() not in [x.lower() for x in existing_aliases]:
                            existing_aliases.append(a.strip())
                # If candidate name was an alias/variant of the canonical name
                if c_name.lower() != matched.get("name", "").lower():
                    if c_name.lower() not in [x.lower() for x in existing_aliases]:
                        existing_aliases.append(c_name)
                matched["aliases"] = existing_aliases

                # Update appearances
                matched["last_appearance"] = f"Chapter {c_num}"

                # Update status transition if reported
                if item.get("status"):
                    record_character_status_transition(
                        matched,
                        new_status=item["status"],
                        chapter_num=c_num,
                        evidence_text=content[:250],
                        status_type="explicit"
                    )

                # Merge personality traits
                new_traits = item.get("personality_traits", [])
                if new_traits:
                    cur_traits = matched.get("personality_traits", [])
                    for t in new_traits:
                        if t and t not in cur_traits:
                            cur_traits.append(t)
                    matched["personality_traits"] = cur_traits

                # Retain physical description or enrich if empty
                if not matched.get("physical_description") and item.get("physical_description"):
                    matched["physical_description"] = item["physical_description"]

                db_service.save_character(matched)
                if matched["id"] not in [u["id"] for u in updated_chars]:
                    updated_chars.append(matched)
            else:
                # Create a new canonical character card
                char_id = f"char_{uuid.uuid4().hex[:10]}"
                raw_status = item.get("status", "Alive")
                init_history = [{
                    "status": raw_status,
                    "chapter": c_num,
                    "evidence": f"Introduced in Chapter {c_num}",
                    "type": "explicit"
                }]

                new_char = {
                    "id": char_id,
                    "story_id": story_id,
                    "name": c_name,
                    "alias": item.get("alias", ""),
                    "aliases": [a for a in c_aliases if a.lower() != c_name.lower()],
                    "age": str(item.get("age", "")),
                    "role": item.get("role", "Supporting"),
                    "personality_traits": item.get("personality_traits", ["Determined"]),
                    "physical_description": item.get("physical_description", ""),
                    "backstory": item.get("backstory", f"Introduced in Chapter {c_num}."),
                    "motivations": item.get("motivations", ""),
                    "goals": item.get("goals", ""),
                    "fears": item.get("fears", ""),
                    "secrets": item.get("secrets", ""),
                    "current_status": raw_status,
                    "status_history": init_history,
                    "first_appearance": f"Chapter {c_num}",
                    "last_appearance": f"Chapter {c_num}",
                    "relationships": [],
                    "evidence": [{"chapter": c_num, "text": content[:200]}],
                    "confidence": 1.0
                }
                db_service.save_character(new_char)
                newly_added.append(new_char)
            
    all_current = db_service.get_characters(story_id)
    return {
        "synced_count": len(newly_added),
        "updated_count": len(updated_chars),
        "new_characters": newly_added,
        "characters": all_current
    }

@router.post("/api/stories/{story_id}/characters", response_model=Character)
def create_character(story_id: str, data: CharacterCreate):
    data_dict = data.dict()
    data_dict["story_id"] = story_id
    data_dict["id"] = f"char_{uuid.uuid4().hex[:10]}"
    return db_service.save_character(data_dict)

@router.get("/api/characters/{character_id}", response_model=Character)
def get_character(character_id: str):
    char = db_service.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character dossier not found")
    return char

@router.put("/api/characters/{character_id}", response_model=Character)
def update_character(character_id: str, data: CharacterUpdate):
    existing = db_service.get_character(character_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Character dossier not found")
    update_data = {k: v for k, v in data.dict().items() if v is not None}
    existing.update(update_data)
    return db_service.save_character(existing)

@router.delete("/api/characters/{character_id}")
def delete_character(character_id: str):
    success = db_service.delete_character(character_id)
    return {"success": success, "deleted_id": character_id}

@router.post("/api/characters/generate")
def generate_ai_character(params: Dict[str, Any]):
    genre = params.get("genre", "Mystery")
    role = params.get("role", "Protagonist")
    theme = params.get("theme", "")
    profile = ai_service.generate_character(genre, role, theme)
    return {"character": profile}

@router.get("/api/stories/{story_id}/board")
def get_story_board(story_id: str):
    board = db_service.get_board(story_id)
    if not board:
        return {"story_id": story_id, "positions": {}, "connections": []}
    return board

@router.put("/api/stories/{story_id}/board")
def update_story_board(story_id: str, board_data: Dict[str, Any]):
    return db_service.save_board(story_id, board_data)

def deduplicate_character_entities_helper(chars: List[Dict[str, Any]], board: Dict[str, Any]):
    """
    Pure in-memory deduplication and board remapping helper for character entities.
    Consolidates variant duplicate records into the fullest canonical name.
    Preserves all unique character information, appearances, traits, status histories,
    and board relationships losslessly.
    """
    chars_copy = [dict(c) for c in chars]
    # Sort characters with longest full names first so canonical identity anchors to the complete name
    chars_copy.sort(key=lambda c: len(c.get("name", "")), reverse=True)

    canonical_list = []
    alias_to_canonical = {}
    purged_ids = []

    for c in chars_copy:
        name = c.get("name", "").strip()
        aliases = c.get("aliases") or []
        if c.get("alias") and c.get("alias") not in aliases:
            aliases.append(c.get("alias"))

        # Find matching canonical character already selected
        matched_canon = None
        for cc in canonical_list:
            # Prevent self-matching
            if cc.get("id") == c.get("id"):
                continue
            if resolve_character_identity(name, aliases, [cc]):
                matched_canon = cc
                break

        if matched_canon:
            alias_to_canonical[c["id"]] = matched_canon["id"]
            purged_ids.append(c["id"])

            # 1. Merge and deduplicate aliases
            ex_aliases = list(matched_canon.get("aliases") or [])
            for a in [name] + aliases:
                clean_a = a.strip()
                if clean_a and clean_a.lower() != matched_canon["name"].lower() and clean_a.lower() not in [x.lower() for x in ex_aliases]:
                    ex_aliases.append(clean_a)
            matched_canon["aliases"] = ex_aliases

            # 2. Merge personality traits
            cur_traits = list(matched_canon.get("personality_traits") or [])
            for t in c.get("personality_traits") or []:
                if t and t not in cur_traits:
                    cur_traits.append(t)
            matched_canon["personality_traits"] = cur_traits

            # 3. Merge Backstory & Descriptions without overwriting
            c_desc = c.get("physical_description", "").strip()
            if c_desc:
                if not matched_canon.get("physical_description"):
                    matched_canon["physical_description"] = c_desc
                elif c_desc.lower() not in matched_canon["physical_description"].lower():
                    matched_canon["physical_description"] = f"{matched_canon['physical_description']}; {c_desc}"

            c_back = c.get("backstory", "").strip()
            if c_back:
                if not matched_canon.get("backstory"):
                    matched_canon["backstory"] = c_back
                elif c_back.lower() not in matched_canon["backstory"].lower():
                    matched_canon["backstory"] = f"{matched_canon['backstory']} {c_back}"

            # 4. Preserve portrait if missing on canonical
            if not matched_canon.get("portrait_url") and c.get("portrait_url"):
                matched_canon["portrait_url"] = c["portrait_url"]

            # 5. Merge Motivations, Goals, Fears, Secrets
            for field in ["motivations", "goals", "fears", "secrets"]:
                c_val = c.get(field, "").strip()
                if c_val:
                    if not matched_canon.get(field):
                        matched_canon[field] = c_val
                    elif c_val.lower() not in matched_canon[field].lower():
                        matched_canon[field] = f"{matched_canon[field]}; {c_val}"

            # 6. Merge Chapter Appearances
            # Preserve earliest first_appearance and latest last_appearance
            c_first = c.get("first_appearance")
            if c_first and not matched_canon.get("first_appearance"):
                matched_canon["first_appearance"] = c_first
            c_last = c.get("last_appearance")
            if c_last and not matched_canon.get("last_appearance"):
                matched_canon["last_appearance"] = c_last

            ex_apps = list(matched_canon.get("appearances") or [])
            for app_item in (c.get("appearances") or []):
                if app_item not in ex_apps:
                    ex_apps.append(app_item)
            matched_canon["appearances"] = ex_apps

            # Update latest status if duplicate had newer appearance
            if c.get("status") and c.get("status") != "Unknown":
                matched_canon["status"] = c["status"]

            # 7. Merge Status History Provenance
            ex_hist = list(matched_canon.get("status_history") or [])
            ex_keys = {(h.get("chapter"), h.get("status", "").lower()) for h in ex_hist}
            for h in c.get("status_history") or []:
                k = (h.get("chapter"), h.get("status", "").lower())
                if k not in ex_keys:
                    ex_hist.append(h)
                    ex_keys.add(k)
            matched_canon["status_history"] = sorted(ex_hist, key=lambda x: x.get("chapter", 0))

            # 8. Merge Relationships recorded inside the character object
            ex_rels = list(matched_canon.get("relationships") or [])
            for r in c.get("relationships") or []:
                r_target = alias_to_canonical.get(r.get("target_id") or r.get("id"), r.get("target_id") or r.get("id"))
                r_target_name = r.get("target_name")
                if (r_target and r_target != matched_canon["id"]) or (r_target_name and r_target_name != matched_canon["name"]):
                    if not any(
                        (x.get("target_id") and x.get("target_id") == r_target) or
                        (x.get("target_name") and x.get("target_name") == r_target_name)
                        for x in ex_rels
                    ):
                        r_copy = dict(r)
                        if r_target:
                            r_copy["target_id"] = r_target
                        ex_rels.append(r_copy)
            matched_canon["relationships"] = ex_rels

        else:
            canonical_list.append(c)
            alias_to_canonical[c["id"]] = c["id"]

    board_copy = dict(board)
    old_positions = board_copy.get("positions", {})
    new_positions = {}
    for cid, pos in old_positions.items():
        canon_id = alias_to_canonical.get(cid, cid)
        if canon_id in [c["id"] for c in canonical_list] and canon_id not in new_positions:
            new_positions[canon_id] = pos

    old_conns = board_copy.get("connections", [])
    new_conns = []
    seen_conn_keys = set()
    canon_ids = {c["id"] for c in canonical_list}
    for conn in old_conns:
        s_id = alias_to_canonical.get(conn.get("sourceId"), conn.get("sourceId"))
        t_id = alias_to_canonical.get(conn.get("targetId"), conn.get("targetId"))
        if s_id and t_id and s_id != t_id and s_id in canon_ids and t_id in canon_ids:
            direction = conn.get("direction", "bidirectional")
            r_type = conn.get("type", "allies")
            if direction == "directional":
                k = (s_id, t_id, r_type)
            else:
                k = tuple(sorted([s_id, t_id])) + (r_type,)
            if k not in seen_conn_keys:
                seen_conn_keys.add(k)
                conn_copy = dict(conn)
                conn_copy["sourceId"] = s_id
                conn_copy["targetId"] = t_id
                new_conns.append(conn_copy)

    board_copy["positions"] = new_positions
    board_copy["connections"] = new_conns
    return canonical_list, purged_ids, board_copy

@router.post("/api/stories/{story_id}/characters/deduplicate")
def deduplicate_story_characters(story_id: str):
    """
    Dossier Hygiene & Canonical Entity Consolidation:
    Identifies duplicate characters (e.g., 'Julian' vs 'Julian Calder', 'Lullu' vs 'Lullu Dickwala')
    in the story's archive, merges their aliases, appearances, and traits into the canonical character,
    rewrites all board positions and connection IDs to the canonical ID, and purges redundant duplicate cards.
    Atomic and transaction-safe.
    """
    story = db_service.get_story(story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")

    chars = db_service.get_characters(story_id)
    if not chars:
        return {"canonical_count": 0, "purged_count": 0, "characters": []}

    board = db_service.get_board(story_id) or {"story_id": story_id, "positions": {}, "connections": []}
    canonical_list, purged_ids, updated_board = deduplicate_character_entities_helper(chars, board)

    # Batch consolidate atomically
    db_service.batch_consolidate_characters(
        story_id=story_id,
        canonical_chars=canonical_list,
        purged_ids=purged_ids,
        board_data=updated_board
    )

    all_current = db_service.get_characters(story_id)
    return {
        "success": True,
        "story_id": story_id,
        "canonical_count": len(all_current),
        "purged_count": len(purged_ids),
        "characters": all_current,
        "board": updated_board
    }

@router.post("/api/stories/{story_id}/board/auto-connect")
def auto_connect_story_board_relationships(story_id: str):
    """
    AI Story Relationship Analyzer & Auto-String Connector:
    Analyzes all chapters and characters in the story, determines authentic interpersonal
    dynamics and relationships, and automatically ties red threads / adjusts connections
    on the dossier board.
    """
    story = db_service.get_story(story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")

    # First, run deduplication to ensure connections attach strictly to canonical characters
    dedupe_res = deduplicate_story_characters(story_id)
    characters = dedupe_res["characters"]
    if len(characters) < 2:
        return {
            "success": False,
            "message": "At least 2 characters are required to analyze relationships.",
            "connections": []
        }

    char_id_map = {c["id"]: c for c in characters}

    chapters = db_service.get_chapters(story_id)
    existing_board = dedupe_res.get("board") or db_service.get_board(story_id) or {"story_id": story_id, "positions": {}, "connections": []}
    existing_conns = existing_board.get("connections", [])

    # Run AI Relationship Analysis over the entire story canon
    discovered_connections = ai_service.analyze_story_relationships_and_connections(
        characters=characters,
        chapters=chapters,
        story=story
    )

    # Filter out any legacy or invalid connections where sourceId or targetId is missing or unknown
    valid_existing_conns = []
    for c in existing_conns:
        s_id = c.get("sourceId")
        t_id = c.get("targetId")
        if s_id in char_id_map and t_id in char_id_map and s_id != t_id:
            # Enrich sourceName and targetName if missing
            if not c.get("sourceName"):
                c["sourceName"] = char_id_map[s_id].get("name", "")
            if not c.get("targetName"):
                c["targetName"] = char_id_map[t_id].get("name", "")
            valid_existing_conns.append(c)

    def get_conn_type_category(t: str) -> str:
        t_clean = (t or "allies").lower()
        if t_clean in ["romance", "lovers", "romantic"]:
            return "romance"
        if t_clean in ["friends", "friendship"]:
            return "friends"
        if t_clean in ["killer_victim", "killer", "victim", "murderer"]:
            return "killer_victim"
        if t_clean in ["enemies", "rivals", "enemy", "rival"]:
            return "enemies"
        if t_clean in ["family", "parent_child", "sibling"]:
            return "family"
        return t_clean

    # Merge with existing connections without overwriting author custom notes if already tied
    merged_connections = list(valid_existing_conns)
    existing_keys = set()
    for c in valid_existing_conns:
        s_id = c.get("sourceId")
        t_id = c.get("targetId")
        if s_id and t_id:
            direction = c.get("direction", "bidirectional")
            r_type = c.get("type", "allies")
            cat = get_conn_type_category(r_type)
            if direction == "directional":
                existing_keys.add((s_id, t_id, cat))
            else:
                existing_keys.add(tuple(sorted([s_id, t_id])) + (cat,))

    added_count = 0
    updated_count = 0
    for disc in discovered_connections:
        s_id = disc["sourceId"]
        t_id = disc["targetId"]
        direction = disc.get("direction", "bidirectional")
        r_type = disc.get("type", "allies")
        cat = get_conn_type_category(r_type)
        
        if direction == "directional":
            identity_key = (s_id, t_id, cat)
        else:
            identity_key = tuple(sorted([s_id, t_id])) + (cat,)

        if identity_key not in existing_keys:
            merged_connections.append(disc)
            existing_keys.add(identity_key)
            added_count += 1
        else:
            # If relationship already exists, update label/notes if new one is richer
            for idx, ex in enumerate(merged_connections):
                ex_s = ex.get("sourceId")
                ex_t = ex.get("targetId")
                ex_dir = ex.get("direction", "bidirectional")
                ex_type = ex.get("type", "allies")
                ex_cat = get_conn_type_category(ex_type)
                if ex_dir == "directional":
                    ex_key = (ex_s, ex_t, ex_cat)
                else:
                    ex_key = tuple(sorted([ex_s, ex_t])) + (ex_cat,) if ex_s and ex_t else None

                if ex_key == identity_key:
                    if disc.get("label") and (len(disc["label"]) > len(ex.get("label", "")) or ex.get("label", "").lower() in ["connected", "unknown connection", "friends"]):
                        merged_connections[idx]["label"] = disc["label"]
                    merged_connections[idx]["direction"] = direction
                    if disc.get("notes") and not merged_connections[idx].get("notes"):
                        merged_connections[idx]["notes"] = disc["notes"]
                    if disc.get("evidence_quote") and not merged_connections[idx].get("evidence_quote"):
                        merged_connections[idx]["evidence_quote"] = disc["evidence_quote"]
                    if disc.get("type"):
                        merged_connections[idx]["type"] = disc["type"]
                    updated_count += 1
                    break

    existing_board["connections"] = merged_connections
    db_service.save_board(story_id, existing_board)

    return {
        "success": True,
        "story_id": story_id,
        "added_count": added_count,
        "updated_count": updated_count,
        "total_connections": len(merged_connections),
        "connections": merged_connections
    }

@router.get("/api/stories/{story_id}/culprit")
def get_story_culprit(story_id: str):
    culprit = db_service.get_culprit(story_id)
    if not culprit:
        return {"story_id": story_id, "culprit_id": None}
    return culprit

@router.put("/api/stories/{story_id}/culprit")
def set_story_culprit(story_id: str, data: Dict[str, Any]):
    culprit_id = data.get("culprit_id")
    if culprit_id:
        char = db_service.get_character(culprit_id)
        if not char:
            raise HTTPException(status_code=404, detail="Character not found")
    return db_service.save_culprit(story_id, {"culprit_id": culprit_id})

@router.delete("/api/stories/{story_id}/culprit")
def remove_story_culprit(story_id: str):
    db_service.delete_culprit(story_id)
    return {"success": True, "story_id": story_id}
