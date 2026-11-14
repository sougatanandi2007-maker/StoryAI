"""
StoryWeave AI - Story-Aware Writing Assistant Service

Provides consultative author feedback on:
- Characters (behaviour, traits, consistency, relationships)
- Plot (pacing, scene tension, plot holes, stakes — never inventing future plot)
- Dialogue (voice, rhythm, subtext, naturalness)
- Writing Style (pacing, sensory details, descriptive precision)
- Continuity (canon alignment, world rule constraints)

Strictly adheres to author sovereignty, optionality, grounding in existing chapters,
and prompt injection defense.
"""

from typing import Optional, List, Dict, Any
from fastapi import Depends

from app.core.logging import get_logger
from app.core.exceptions import StoryNotFoundException, ChapterNotFoundException, ValidationException
from app.database.repositories.base import StoryRepository, ChapterRepository, CharacterRepository
from app.database.repositories.dependencies import (
    get_story_repository,
    get_chapter_repository,
    get_character_repository,
)
from app.services.retrieval_service import RetrievalService, get_retrieval_service
from app.services.context_builder import ContextBuilder, get_context_builder
from app.services.gemini_service import GeminiService, get_gemini_service
from app.schemas.assistant import AssistantChatRequest, AssistantChatResponse, FeedbackType

logger = get_logger("service.assistant")


class WritingAssistantService:
    """Orchestrates story-grounded writing assistance and advice."""

    def __init__(
        self,
        story_repo: StoryRepository,
        chapter_repo: ChapterRepository,
        character_repo: CharacterRepository,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilder,
        gemini_service: GeminiService,
    ):
        self._story_repo = story_repo
        self._chapter_repo = chapter_repo
        self._character_repo = character_repo
        self._retrieval_service = retrieval_service
        self._context_builder = context_builder
        self._gemini_service = gemini_service

    async def provide_feedback(
        self,
        request: AssistantChatRequest,
        user_id: str,
    ) -> AssistantChatResponse:
        """
        Evaluate author request and generate story-grounded creative writing advice.

        Guarantees:
        - Strict user and story isolation.
        - Grounded context from ChromaDB memories and character profiles.
        - Advisory boundaries: no autonomous plot creation, no manuscript modification.
        """
        if not request.message or not request.message.strip():
            raise ValidationException("Message query cannot be empty.")

        # 1. Fetch story and verify ownership (Story Isolation)
        story = await self._story_repo.get_by_id(story_id=request.story_id, user_id=user_id)
        if not story:
            raise StoryNotFoundException(story_id=request.story_id)

        # 2. If chapter_id provided, verify chapter ownership and extract details
        chapter_info: Optional[str] = None
        chapter_number: Optional[int] = None
        if request.chapter_id:
            chapter = await self._chapter_repo.get_by_id(chapter_id=request.chapter_id, user_id=user_id)
            if not chapter or chapter.get("story_id") != request.story_id:
                raise ChapterNotFoundException(chapter_id=request.chapter_id)

            chapter_number = chapter.get("chapter_number")
            chapter_info = (
                f"\n=== CHAPTER IN FOCUS ===\n"
                f"Chapter {chapter_number}: '{chapter.get('title', 'Untitled')}'\n"
                f"Current Chapter Manuscript Excerpt:\n\"\"\"{chapter.get('content', '')[:2000]}\"\"\"\n"
            )

        # 3. Retrieve relevant memory chunks from ChromaDB
        query_text = f"{request.message.strip()} {request.passage.strip() if request.passage else ''}".strip()
        chunks = await self._retrieval_service.retrieve_relevant_memories(
            query=query_text,
            story_id=request.story_id,
            user_id=user_id,
            top_k=5,
            min_relevance=0.15,
        )

        # 4. Retrieve matching character profiles for context
        matching_characters: List[Dict[str, Any]] = []
        try:
            story_chars = await self._character_repo.list_by_story(story_id=request.story_id, user_id=user_id)
            query_lower = query_text.lower()
            for c in story_chars:
                name = c.get("name", "").lower()
                if name and name in query_lower:
                    matching_characters.append(c)
                    continue
                for a in c.get("aliases", []):
                    if a and a.lower() in query_lower:
                        matching_characters.append(c)
                        break

            # If no direct name hit, include existing characters (up to 4) to anchor character advice
            if not matching_characters and story_chars and request.feedback_type in (FeedbackType.CHARACTER, FeedbackType.DIALOGUE):
                matching_characters = story_chars[:4]
        except Exception as e:
            logger.warning(f"Error gathering character profiles for assistant context: {e}")

        # 5. Build grounded context using ContextBuilder
        built_context = self._context_builder.build_context(
            chunks=chunks,
            story_context=story,
            character_context=matching_characters,
        )

        # 6. Call Gemini advisor
        ai_result = await self._gemini_service.generate_assistant_feedback(
            message=request.message,
            feedback_type=request.feedback_type.value,
            context=built_context.formatted_context,
            passage=request.passage,
            chapter_info=chapter_info,
        )

        # 7. Collect referenced chapter numbers
        referenced_chapters = set()
        if chapter_number:
            referenced_chapters.add(chapter_number)
        for chunk in built_context.included_chunks:
            referenced_chapters.add(chunk.chapter_number)
        for c in ai_result.get("relevant_chapters", []):
            referenced_chapters.add(c)

        # Check if context was genuinely insufficient
        insufficient = (
            ai_result.get("insufficient_context", False)
            or (len(built_context.included_chunks) == 0 and not matching_characters and not chapter_info)
        )

        logger.info(
            f"Provided {request.feedback_type.value} feedback for story '{request.story_id}' "
            f"({len(ai_result.get('suggestions', []))} suggestions, {len(referenced_chapters)} chapters referenced)."
        )

        return AssistantChatResponse(
            feedback=ai_result.get("feedback", ""),
            story_id=request.story_id,
            feedback_type=request.feedback_type,
            suggestions=ai_result.get("suggestions", []),
            relevant_chapters=sorted(list(referenced_chapters)),
            insufficient_context=insufficient,
            clarifying_questions=ai_result.get("clarifying_questions", []),
            model_used=self._gemini_service.model_name,
        )


# Singleton instance
_assistant_service: Optional[WritingAssistantService] = None


def get_writing_assistant_service(
    story_repo: StoryRepository = Depends(get_story_repository),
    chapter_repo: ChapterRepository = Depends(get_chapter_repository),
    character_repo: CharacterRepository = Depends(get_character_repository),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    context_builder: ContextBuilder = Depends(get_context_builder),
    gemini_service: GeminiService = Depends(get_gemini_service),
) -> WritingAssistantService:
    """Dependency provider for WritingAssistantService."""
    global _assistant_service
    if _assistant_service is None:
        _assistant_service = WritingAssistantService(
            story_repo=story_repo,
            chapter_repo=chapter_repo,
            character_repo=character_repo,
            retrieval_service=retrieval_service,
            context_builder=context_builder,
            gemini_service=gemini_service,
        )
    return _assistant_service
