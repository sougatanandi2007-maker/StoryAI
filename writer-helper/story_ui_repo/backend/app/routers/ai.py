from fastapi import APIRouter
from typing import Dict, Any
from ..services.ai_service import ai_service
from ..services.vector_service import vector_service
from ..services.db_service import db_service

router = APIRouter(prefix="/api/ai", tags=["AI Co-writer"])

@router.post("/cowriter")
def cowriter_assistance(payload: Dict[str, Any]):
    story_id = payload.get("story_id", "")
    chapter_id = payload.get("chapter_id", "")
    action = payload.get("action", "continue")
    content = payload.get("content", "")
    chapter_title = payload.get("chapter_title", "Active Scene")
    character_notes = payload.get("character_notes", "")
    instruction = payload.get("instruction", "")
    
    # 1. Fetch full story universe specifications
    story = db_service.get_story(story_id) if story_id else None
    
    # 2. Fetch all chapters in chronological canon
    all_chapters = db_service.get_chapters(story_id) if story_id else []
    
    # 3. Fetch all characters in the universe
    all_characters = db_service.get_characters(story_id) if story_id else []
    
    # 4. Fetch timeline events
    all_events = db_service.get_events(story_id) if story_id else []
    
    # 5. Fetch persistent lore memories
    all_mems = db_service.get_memories(story_id) if story_id else []
    
    # Semantic search across both memories AND prior chapter texts
    query_text = f"{content[-600:]} {instruction}".strip()
    relevant_memories = vector_service.search_memories(
        query=query_text,
        memories=all_mems,
        chapters=all_chapters,
        current_chapter_id=chapter_id,
        top_k=5
    ) if story_id else []
    
    res = ai_service.assist_writing(
        action=action,
        current_content=content,
        chapter_title=chapter_title,
        chapter_id=chapter_id,
        story=story,
        chapters=all_chapters,
        characters=all_characters,
        events=all_events,
        relevant_memories=relevant_memories,
        character_notes=character_notes,
        instruction=instruction
    )
    res["injected_memories_count"] = len(relevant_memories)
    res["injected_memories"] = relevant_memories
    return res
