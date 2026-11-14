"""
StoryWeave AI - Memory Service

Coordinates manuscript chunking, vector embedding generation, ChromaDB vector indexing,
and story/user-isolated semantic memory retrieval.
"""

import math
from typing import Optional, List, Dict, Any
from app.core.logging import get_logger
from app.core.exceptions import (
    StoryNotFoundException,
    ChapterNotFoundException,
    ValidationException,
    MemoryServiceError,
)
from fastapi import Depends
from app.database.chromadb import ChromaDBManager, get_chromadb_manager
from app.database.repositories.base import StoryRepository, ChapterRepository, AnalysisRepository
from app.database.repositories.dependencies import (
    get_story_repository,
    get_chapter_repository,
    get_analysis_repository,
)
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.chunking_service import ChunkingService, get_chunking_service
from app.schemas.chapter import ProcessingStatus
from app.schemas.memory import (
    MemoryChunk,
    IndexingReport,
    MemorySearchResponse,
    MemorySearchResult,
    StoryMemoryStatusResponse,
    StoredMemoryItem,
    PaginatedMemoryResponse,
)

logger = get_logger("service.memory")


class MemoryService:
    """Service managing persistent vector memories for story universes."""

    def __init__(
        self,
        chroma_manager: ChromaDBManager,
        embedding_service: EmbeddingService,
        chunking_service: ChunkingService,
        story_repo: StoryRepository,
        chapter_repo: ChapterRepository,
        analysis_repo: AnalysisRepository,
    ):
        self._chroma = chroma_manager
        self._embedding_service = embedding_service
        self._chunking_service = chunking_service
        self._story_repo = story_repo
        self._chapter_repo = chapter_repo
        self._analysis_repo = analysis_repo

    async def index_chapter(
        self,
        chapter_id: str,
        user_id: str,
        require_analysis: bool = False,
    ) -> IndexingReport:
        """
        Chunk, embed, and index a chapter manuscript into persistent ChromaDB memory.

        Workflow:
        1. Validate chapter and parent story exist and belong to user.
        2. Ensure chapter analysis has been completed (if require_analysis=True).
        3. Divide manuscript into natural semantic chunks using ChunkingService.
        4. Generate dense vector embeddings using Sentence Transformers.
        5. Store chunks, vectors, and source metadata in ChromaDB using deterministic IDs.
        6. Return indexing report.
        """
        # 1. Fetch chapter
        chapter = await self._chapter_repo.get_by_id(chapter_id=chapter_id, user_id=user_id)
        if not chapter:
            raise ChapterNotFoundException(chapter_id=chapter_id)

        story_id = chapter["story_id"]
        story = await self._story_repo.get_by_id(story_id=story_id, user_id=user_id)
        if not story:
            raise StoryNotFoundException(story_id=story_id)

        # 2. Check analysis status if required
        if require_analysis:
            report = await self._analysis_repo.get_by_chapter_id(chapter_id=chapter_id, user_id=user_id)
            if not report:
                raise ValidationException(
                    f"Chapter '{chapter['title']}' must complete AI analysis before memory indexing. "
                    f"Call POST /api/v1/chapters/{chapter_id}/analyze first."
                )

        content = chapter.get("content", "")
        if not content or not content.strip():
            raise ValidationException("Chapter content is empty; cannot index.")

        # 3. Create semantic chunks
        chunks: List[MemoryChunk] = self._chunking_service.chunk_chapter(
            chapter_id=chapter_id,
            story_id=story_id,
            user_id=user_id,
            chapter_number=chapter["chapter_number"],
            content=content,
        )

        if not chunks:
            raise ValidationException("Failed to generate any chunks from chapter content.")

        # 4. Generate batch embeddings
        chunk_texts = [c.text for c in chunks]
        chunk_ids = [c.id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # 4b. If chapter has a generated summary, also index it as a separate memory layer
        summary_text = chapter.get("summary")
        if summary_text and summary_text.strip():
            summary_id = f"summary_{chapter_id}"
            chunk_ids.append(summary_id)
            chunk_texts.append(f"Chapter {chapter['chapter_number']} Summary: {summary_text.strip()}")
            metadatas.append({
                "chapter_id": chapter_id,
                "story_id": story_id,
                "user_id": user_id,
                "chapter_number": chapter["chapter_number"],
                "chunk_index": -1,
                "memory_type": "chapter_summary",
                "source": f"Chapter {chapter['chapter_number']} Summary",
            })

        embeddings = self._embedding_service.embed_batch(chunk_texts, normalize=True)

        # 5. Upsert into ChromaDB
        self._chroma.upsert_chunks(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            f"Indexed {len(chunk_ids)} memory items (including summary: {bool(summary_text)}) "
            f"for chapter {chapter['chapter_number']} ('{chapter['title']}') into story '{story_id}'."
        )

        return IndexingReport(
            chapter_id=chapter_id,
            story_id=story_id,
            chunks_indexed=len(chunk_ids),
            word_count=chapter.get("word_count", 0),
            status="completed",
            message=f"Successfully indexed {len(chunk_ids)} semantic memory chunks into persistent vector memory.",
        )

    async def search_memories(
        self,
        query: str,
        user_id: str,
        story_id: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> MemorySearchResponse:
        """
        Execute semantic similarity search across memories for a story universe.

        Strictly enforces user_id and story_id isolation.
        """
        if not query or not query.strip():
            raise ValidationException("Search query cannot be empty.")

        # Verify story exists and belongs to user
        story_exists = await self._story_repo.exists(story_id=story_id, user_id=user_id)
        if not story_exists:
            raise StoryNotFoundException(story_id=story_id)

        # Generate query embedding
        query_vector = self._embedding_service.embed_text(query.strip(), normalize=True)

        # Search ChromaDB
        raw_matches = self._chroma.search(
            query_embedding=query_vector,
            user_id=user_id,
            story_id=story_id,
            top_k=top_k,
            memory_type=memory_type,
        )

        results = [
            MemorySearchResult(
                id=m["id"],
                text=m["text"],
                metadata=m["metadata"],
                similarity_score=m["similarity_score"],
                distance=m["distance"],
            )
            for m in raw_matches
        ]

        return MemorySearchResponse(
            query=query,
            story_id=story_id,
            total_found=len(results),
            results=results,
        )

    async def get_memories(
        self,
        story_id: str,
        user_id: str,
        chapter_id: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
    ) -> PaginatedMemoryResponse:
        """Retrieve paginated stored memories for a story."""
        story_exists = await self._story_repo.exists(story_id=story_id, user_id=user_id)
        if not story_exists:
            raise StoryNotFoundException(story_id=story_id)

        skip = (page - 1) * limit
        items_raw, total = self._chroma.get_memories_for_story(
            story_id=story_id,
            user_id=user_id,
            chapter_id=chapter_id,
            limit=limit,
            offset=skip,
        )

        items = [
            StoredMemoryItem(
                id=m["id"],
                text=m["text"],
                metadata=m["metadata"],
            )
            for m in items_raw
        ]

        total_pages = math.ceil(total / limit) if total > 0 else 1

        return PaginatedMemoryResponse(
            items=items,
            total=total,
            page=page,
            limit=limit,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )

    async def get_memory_status(self, story_id: str, user_id: str) -> StoryMemoryStatusResponse:
        """Get indexing statistics and status for a story universe."""
        story_exists = await self._story_repo.exists(story_id=story_id, user_id=user_id)
        if not story_exists:
            raise StoryNotFoundException(story_id=story_id)

        # Retrieve all memory metadata for this story
        items_raw, total = self._chroma.get_memories_for_story(
            story_id=story_id,
            user_id=user_id,
            limit=10000,
            offset=0,
        )

        indexed_chapters = sorted(
            list({m["metadata"]["chapter_id"] for m in items_raw if "chapter_id" in m.get("metadata", {})})
        )

        return StoryMemoryStatusResponse(
            story_id=story_id,
            total_chunks=total,
            indexed_chapters_count=len(indexed_chapters),
            indexed_chapter_ids=indexed_chapters,
            is_indexed=total > 0,
        )

    async def delete_chapter_memories(self, chapter_id: str, user_id: str) -> None:
        """Remove all memory chunks for a chapter."""
        self._chroma.delete_by_chapter(chapter_id=chapter_id, user_id=user_id)

    async def delete_story_memories(self, story_id: str, user_id: str) -> None:
        """Remove all memory chunks for an entire story."""
        self._chroma.delete_by_story(story_id=story_id, user_id=user_id)


# Dependency provider
def get_memory_service(
    chroma_manager: ChromaDBManager = Depends(get_chromadb_manager),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    chunking_service: ChunkingService = Depends(get_chunking_service),
    story_repo: StoryRepository = Depends(get_story_repository),
    chapter_repo: ChapterRepository = Depends(get_chapter_repository),
    analysis_repo: AnalysisRepository = Depends(get_analysis_repository),
) -> MemoryService:
    """Dependency provider for MemoryService."""
    return MemoryService(
        chroma_manager=chroma_manager,
        embedding_service=embedding_service,
        chunking_service=chunking_service,
        story_repo=story_repo,
        chapter_repo=chapter_repo,
        analysis_repo=analysis_repo,
    )
