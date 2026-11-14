from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any

from ..models import Memory, MemoryCreate, MemoryUpdate
from ..services.db_service import db_service
from ..services.vector_service import vector_service
from ..services.ai_service import ai_service
import uuid
import re

router = APIRouter(tags=["Universe Memory"])

@router.get("/api/stories/{story_id}/memories", response_model=List[Memory])
def get_story_memories(story_id: str, category: Optional[str] = Query(None)):
    mems = db_service.get_memories(story_id, category)
    # If no explicit user lore memories exist yet, synthesize passage fragments directly from indexed chapters
    if not mems:
        chapters = db_service.get_chapters(story_id)
        passage_mems = []
        for ch in chapters:
            c_num = ch.get("chapter_number", 1)
            c_title = ch.get("title", f"Chapter {c_num}")
            c_id = ch.get("id", "chap")
            if ch.get("summary"):
                passage_mems.append({
                    "id": f"mem_summary_{c_id}",
                    "story_id": story_id,
                    "title": f"Chapter {c_num}: {c_title} Summary",
                    "description": ch["summary"],
                    "category": "important_events",
                    "source_chapter": f"Chapter {c_num}",
                    "importance": "High"
                })
            chunks = vector_service.chunk_manuscript(c_id, ch.get("content", ""), words_per_chunk=300, overlap_words=30)
            for chk in chunks[:2]:
                passage_mems.append({
                    "id": chk["chunk_id"],
                    "story_id": story_id,
                    "title": f"Chapter {c_num}: {c_title} (Passage {chk['index']+1})",
                    "description": chk["text"],
                    "category": "story_specific" if not category else category,
                    "source_chapter": f"Chapter {c_num}",
                    "importance": "Medium"
                })
        if category:
            passage_mems = [m for m in passage_mems if m.get("category") == category]
        return passage_mems
    return mems

@router.post("/api/stories/{story_id}/memories", response_model=Memory)
def create_memory(story_id: str, data: MemoryCreate):
    data_dict = data.dict()
    data_dict["story_id"] = story_id
    data_dict["id"] = f"mem_{uuid.uuid4().hex[:10]}"
    return db_service.save_memory(data_dict)

@router.put("/api/memories/{memory_id}", response_model=Memory)
def update_memory(memory_id: str, data: MemoryUpdate):
    for s in db_service.get_stories():
        mems = db_service.get_memories(s["id"])
        match = [m for m in mems if m["id"] == memory_id]
        if match:
            existing = match[0]
            update_data = {k: v for k, v in data.dict().items() if v is not None}
            existing.update(update_data)
            return db_service.save_memory(existing)
    raise HTTPException(status_code=404, detail="Memory fragment not found")

@router.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: str):
    success = db_service.delete_memory(memory_id)
    return {"success": success, "deleted_id": memory_id}

@router.post("/api/stories/{story_id}/memories/search")
def search_memories(story_id: str, payload: Dict[str, Any]):
    query = payload.get("query", "")
    top_k = int(payload.get("top_k", 6))
    current_chapter_id = payload.get("current_chapter_id")
    all_mems = db_service.get_memories(story_id)
    all_chapters = db_service.get_chapters(story_id)
    results = vector_service.search_memories(
        query=query,
        memories=all_mems,
        chapters=all_chapters,
        current_chapter_id=current_chapter_id,
        top_k=top_k,
        story_id=story_id
    )
    return {"query": query, "results": results, "count": len(results)}

@router.get("/api/memory/pipeline")
def get_pipeline_details():
    return vector_service.get_pipeline_explanation()

@router.post("/api/stories/{story_id}/memories/reindex")
def reindex_story_universe(story_id: str):
    """
    Forces complete semantic re-indexing of all chapters in the story into ChromaDB.
    """
    story = db_service.get_story(story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    chapters = db_service.get_chapters(story_id)
    indexed_count = vector_service.sync_story_chapters(story_id, chapters)
    return {
        "success": True,
        "story_id": story_id,
        "chapters_indexed": len(chapters),
        "indexed_chapters": len(chapters),
        "total_chunks": indexed_count,
        "message": f"Indexed {len(chapters)} chapter(s) into {indexed_count} semantic chunks.",
        "engine": "ChromaDB + all-MiniLM-L6-v2"
    }

@router.post("/api/stories/{story_id}/universe-query")
def answer_universe_memory_query(story_id: str, payload: Dict[str, Any]):
    """
    StoryWeave Universe Memory RAG Answer Engine:
    Retrieves evidence across all chapters via ChromaDB semantic search and generates
    an authentic, strictly grounded response with exact chapter citations and uncertainty signaling.
    """
    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    story = db_service.get_story(story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")

    conversation_history = payload.get("history", [])

    all_chapters = db_service.get_chapters(story_id)
    all_characters = db_service.get_characters(story_id)
    all_mems = db_service.get_memories(story_id)

    # 1. Pronoun and reference resolution using conversation history if needed
    retrieval_query = question
    if conversation_history and len(conversation_history) >= 2:
        last_exchange = f"Q: {conversation_history[-2].get('content', '')} A: {conversation_history[-1].get('content', '')[:100]}"
        # If question contains pronouns like 'he', 'she', 'him', 'her', 'it', 'they', augment query
        if any(re.search(r'\b' + p + r'\b', question.lower()) for p in ["he", "she", "him", "her", "they", "them", "it", "afterward", "afterwards", "then"]):
            retrieval_query = f"{question} {conversation_history[-2].get('content', '')}"

    # 2. Semantic vector retrieval across all chapters and memories
    retrieved_chunks = vector_service.search_memories(
        query=retrieval_query,
        memories=all_mems,
        chapters=all_chapters,
        top_k=8,
        story_id=story_id
    )

    # 3. Assemble Grounded Context and Citations
    context_blocks = []
    citations = []
    seen_chapters = set()

    for idx, chunk in enumerate(retrieved_chunks):
        c_num = chunk.get("chapter_number", 1)
        c_title = chunk.get("title", f"Chapter {c_num}")
        c_src = chunk.get("source_chapter", f"Chapter {c_num}")
        snippet = chunk.get("description", "")
        context_blocks.append(f"[SOURCE: {c_src} | Excerpt {idx+1}]\n{snippet}")

        if c_src not in seen_chapters:
            seen_chapters.add(c_src)
            citations.append({
                "chapter": c_src,
                "title": c_title,
                "excerpt": snippet[:200] + "..." if len(snippet) > 200 else snippet,
                "relevance_score": chunk.get("relevance_score", 0.0)
            })

    # Add character intelligence context if characters mentioned in query
    char_context = []
    q_low = retrieval_query.lower()
    for c in all_characters:
        c_name = c.get("name", "")
        if c_name and c_name.lower() in q_low:
            char_context.append(f"Character {c_name} (Status: {c.get('current_status')}, Role: {c.get('role')}): {c.get('backstory', '')[:200]}")

    formatted_context = "\n\n".join(context_blocks)
    if char_context:
        formatted_context = "=== ESTABLISHED CHARACTER PROFILES ===\n" + "\n".join(char_context) + "\n\n=== RETRIEVED CHAPTER EVIDENCE ===\n" + formatted_context

    # 4. Generate Grounded Answer using Gemini
    system_instruction = (
        "You are StoryWeave Universe Memory, an intelligent assistant that understands the user's fictional universe.\n\n"
        "Answer questions using retrieved manuscript evidence.\n\n"
        "Rules:\n"
        "- Answer the exact question directly.\n"
        "- Synthesize relevant information from multiple chapters when necessary.\n"
        "- Never copy entire manuscript passages as answers.\n"
        "- Never introduce unrelated narrative details.\n"
        "- Never invent characters, events, relationships, dialogue, or motivations.\n"
        "- Distinguish established facts from contextual inferences.\n"
        "- If evidence is insufficient, explicitly say so: 'The available chapters do not establish this information.'\n"
        "- If chapters contradict each other, explain the contradiction.\n"
        "- Resolve character aliases correctly.\n"
        "- Keep direct factual answers concise.\n"
        "- Provide detailed explanations only when requested.\n\n"
        "Retrieved passages are supporting evidence, not the final response.\n"
        "At the end of your response, output a single line formatted as:\n"
        "GROUNDING_STATUS: [HIGH | MEDIUM | LOW | INSUFFICIENT EVIDENCE | CONTRADICTORY]"
    )

    prompt = f"""Story Title: {story.get('title', 'Novel Universe')}
Author's Query: {question}

=== RETRIEVED MANUSCRIPT EVIDENCE ===
{formatted_context}

Provide a direct, natural-language answer to the Author's Query based solely on the retrieved evidence above.
Do not recite the manuscript. Answer the specific question directly.
Remember to conclude with:
GROUNDING_STATUS: [HIGH | MEDIUM | LOW | INSUFFICIENT EVIDENCE | CONTRADICTORY]"""

    ai_answer = None
    explicit_confidence = None
    if formatted_context.strip():
        ai_res = ai_service._call_gemini(prompt, system_instruction=system_instruction)
        if ai_res and ai_res.strip():
            # Check for GROUNDING_STATUS line
            status_match = re.search(r'GROUNDING_STATUS:\s*(HIGH|MEDIUM|LOW|INSUFFICIENT EVIDENCE|CONTRADICTORY)', ai_res, re.IGNORECASE)
            if status_match:
                explicit_confidence = status_match.group(1).upper()
                ai_answer = re.sub(r'GROUNDING_STATUS:\s*(HIGH|MEDIUM|LOW|INSUFFICIENT EVIDENCE|CONTRADICTORY)', '', ai_res, flags=re.IGNORECASE).strip()
            else:
                ai_answer = ai_res.strip()

    # Deterministic fallback answer synthesizer if Gemini is unavailable or rate-limited
    if not ai_answer:
        if not retrieved_chunks:
            ai_answer = "The available chapters do not establish this information, as no matching records were retrieved from the story universe."
            explicit_confidence = "INSUFFICIENT EVIDENCE"
        else:
            # Extract most precise answer sentences matching the query
            stopwords = {'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how', 'the', 'was', 'were', 'is', 'are', 'did', 'does', 'into', 'from', 'with', 'about', 'and', 'that', 'this', 'for', 'in', 'of', 'on', 'at', 'to', 'a', 'an'}
            q_words = [w.lower() for w in re.findall(r'\w+', question) if len(w) > 1 and w.lower() not in stopwords]
            
            best_sentences = []
            best_src = None
            best_score = 0
            
            for chunk in retrieved_chunks:
                src = chunk.get('source_chapter', 'Chapter 1')
                text = chunk.get('description', '')
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for idx, s in enumerate(sentences):
                    s_clean = s.strip()
                    if not s_clean:
                        continue
                    s_lower = s_clean.lower()
                    matches = sum(1 for w in q_words if w in s_lower)
                    if matches > best_score:
                        best_score = matches
                        surrounding = []
                        if idx > 0:
                            surrounding.append(sentences[idx-1].strip())
                        surrounding.append(s_clean)
                        if idx + 1 < len(sentences):
                            surrounding.append(sentences[idx+1].strip())
                        best_sentences = surrounding
                        best_src = src
            
            if best_sentences and best_score >= 1:
                ans_text = " ".join(best_sentences)
                ai_answer = f"{ans_text}\n\nSources:\n- {best_src}"
                explicit_confidence = "HIGH" if best_score >= 3 else "MEDIUM"
            else:
                top_snippet = retrieved_chunks[0].get("description", "")
                top_src = retrieved_chunks[0].get("source_chapter", "Chapter 1")
                ai_answer = f"Based on {top_src}: {top_snippet}\n\nSources:\n- {top_src}"
                explicit_confidence = "LOW"

    # Separate Sources section from main answer if attached
    clean_answer = re.sub(r'\n+\s*\*{0,2}Sources:?\*{0,2}[\s\S]*$', '', ai_answer).strip()

    # Determine confidence and grounded status
    ans_lower = ai_answer.lower()
    is_insufficient = (
        not retrieved_chunks
        or "do not establish" in ans_lower
        or "does not establish" in ans_lower
        or "insufficient information" in ans_lower
        or "insufficient evidence" in ans_lower
        or "cannot find any mention" in ans_lower
        or "no mention" in ans_lower
    )

    if explicit_confidence:
        confidence = explicit_confidence
    elif is_insufficient:
        confidence = "INSUFFICIENT EVIDENCE"
    elif "contradict" in ans_lower:
        confidence = "CONTRADICTORY"
    elif len(retrieved_chunks) >= 2:
        confidence = "HIGH"
    else:
        confidence = "MEDIUM"

    if is_insufficient:
        grounded = False
        if confidence != "CONTRADICTORY":
            confidence = "INSUFFICIENT EVIDENCE"
    else:
        grounded = confidence in {"HIGH", "MEDIUM"}

    return {
        "success": True,
        "story_id": story_id,
        "question": question,
        "resolved_query": retrieval_query,
        "answer": clean_answer,
        "citations": [
            {
                "chapter_id": c.get("chapter_id", ""),
                "chapter_number": c.get("chapter_number", 1),
                "chapter_title": c.get("title", ""),
                "source": c.get("chapter", "Chapter 1"),
                "excerpt": c.get("excerpt", ""),
                "similarity": c.get("relevance_score", 0.8)
            }
            for c in citations
        ],
        "grounded": grounded,
        "confidence": confidence,
        "retrieved_count": len(retrieved_chunks),
        "sources": [c["chapter"] for c in citations]
    }
