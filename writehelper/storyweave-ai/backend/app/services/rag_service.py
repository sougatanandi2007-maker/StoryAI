"""
StoryWeave AI - RAG Generation Service

Orchestrates the complete RAG pipeline:
1. User question validation
2. Story-scoped, isolated memory retrieval (RetrievalService)
3. Structured context construction with budget enforcement (ContextBuilder)
4. Grounded answer generation with prompt injection protection (GeminiService)
5. Source attribution and uncertainty detection
"""

import time
from typing import Optional, List, Dict, Any
from fastapi import Depends
from app.core.logging import get_logger
from app.core.exceptions import StoryNotFoundException, ValidationException
from app.database.repositories.base import StoryRepository, CharacterRepository
from app.database.repositories.dependencies import get_story_repository, get_character_repository
from app.services.retrieval_service import RetrievalService, get_retrieval_service
from app.services.context_builder import ContextBuilder, get_context_builder
from app.services.gemini_service import GeminiService, get_gemini_service
from app.schemas.rag import ChatRequest, ChatResponse, ChatSourceReference

logger = get_logger("service.rag")


class RAGService:
    """Service orchestrating story-scoped memory retrieval and grounded Gemini generation."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilder,
        gemini_service: GeminiService,
        story_repo: StoryRepository,
        character_repo: Optional[CharacterRepository] = None,
    ):
        self._retrieval_service = retrieval_service
        self._context_builder = context_builder
        self._gemini_service = gemini_service
        self._story_repo = story_repo
        self._character_repo = character_repo

    async def answer_question(
        self,
        story_id: str,
        message: str,
        user_id: str,
        top_k: int = 5,
        min_relevance: float = 0.25,
    ) -> ChatResponse:
        """
        Execute the full RAG pipeline for an author's question.

        Enforces:
        - Strict user and story isolation.
        - Grounded response with provenance citations.
        - Proper detection of insufficient context.
        """
        start_time = time.perf_counter()

        if not message or not message.strip():
            raise ValidationException("Message query cannot be empty.")

        # 1. Fetch story and verify ownership
        story = await self._story_repo.get_by_id(story_id=story_id, user_id=user_id)
        if not story:
            raise StoryNotFoundException(story_id=story_id)

        # 2. Retrieve relevant memory chunks from ChromaDB
        chunks = await self._retrieval_service.retrieve_relevant_memories(
            query=message.strip(),
            story_id=story_id,
            user_id=user_id,
            top_k=top_k,
            min_relevance=min_relevance,
        )

        # 2b. Retrieve matching character profiles (Phase 6: Character Intelligence)
        matching_characters: List[Dict[str, Any]] = []
        if self._character_repo:
            try:
                story_chars = await self._character_repo.list_by_story(story_id=story_id, user_id=user_id)
                msg_lower = message.lower()
                for c in story_chars:
                    c_name = c.get("name", "").lower()
                    if c_name and c_name in msg_lower:
                        matching_characters.append(c)
                        continue
                    for a in c.get("aliases", []):
                        if a and a.lower() in msg_lower:
                            matching_characters.append(c)
                            break
            except Exception as char_err:
                logger.warning(f"Error retrieving characters for RAG context: {char_err}")

        # 3. Assemble grounded context
        built_context = self._context_builder.build_context(
            chunks=chunks,
            story_context=story,
            character_context=matching_characters,
        )

        # 4. Generate grounded answer via Gemini
        raw_answer = await self._gemini_service.generate_rag_answer(
            question=message.strip(),
            context=built_context.formatted_context,
        )

        # 5. Detect if answer reflects insufficient context
        answer_text = raw_answer.strip()
        lower_ans = answer_text.lower()
        insufficient = (
            (len(built_context.included_chunks) == 0 and not matching_characters)
            or "insufficient information" in lower_ans
            or "insufficient context" in lower_ans
            or "not enough information" in lower_ans
            or "not mentioned in the provided" in lower_ans
            or "cannot find any mention" in lower_ans
        )

        # 6. Format source citations
        sources: List[ChatSourceReference] = [
            chunk.to_source_reference()
            for chunk in built_context.included_chunks
        ]

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            f"Completed RAG query for story '{story_id}' in {elapsed_ms}ms: "
            f"{len(sources)} sources cited, {len(matching_characters)} characters included, "
            f"insufficient_context={insufficient}."
        )

        return ChatResponse(
            answer=answer_text,
            story_id=story_id,
            sources=sources,
            retrieval_count=len(sources),
            insufficient_context=insufficient,
            model_used=self._gemini_service.model_name,
            processing_time_ms=elapsed_ms,
        )


def get_rag_service(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    context_builder: ContextBuilder = Depends(get_context_builder),
    gemini_service: GeminiService = Depends(get_gemini_service),
    story_repo: StoryRepository = Depends(get_story_repository),
    character_repo: CharacterRepository = Depends(get_character_repository),
) -> RAGService:
    """FastAPI dependency provider for RAGService."""
    return RAGService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        gemini_service=gemini_service,
        story_repo=story_repo,
        character_repo=character_repo,
    )
