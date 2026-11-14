"""
StoryWeave AI - Story Memory Retrieval Service

Coordinates query vectorization, isolated vector search across persistent ChromaDB memories,
relevance scoring, deduplication, and provenance verification.
"""

from typing import List, Dict, Any, Optional
from fastapi import Depends
from app.core.logging import get_logger
from app.core.exceptions import StoryNotFoundException, ValidationException
from app.database.chromadb import ChromaDBManager, get_chromadb_manager
from app.database.repositories.base import StoryRepository
from app.database.repositories.dependencies import get_story_repository
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.schemas.rag import ChatSourceReference

logger = get_logger("service.retrieval")


class RetrievedMemoryChunk:
    """Internal model for a retrieved memory chunk with provenance and relevance metrics."""

    def __init__(
        self,
        id: str,
        text: str,
        metadata: Dict[str, Any],
        similarity_score: float,
        distance: float,
    ):
        self.id = id
        self.text = text
        self.metadata = metadata
        self.similarity_score = similarity_score
        self.distance = distance

    @property
    def chapter_id(self) -> str:
        return str(self.metadata.get("chapter_id", "unknown"))

    @property
    def chapter_number(self) -> int:
        return int(self.metadata.get("chapter_number", 0))

    @property
    def memory_type(self) -> str:
        return str(self.metadata.get("memory_type", "manuscript_chunk"))

    @property
    def source_text(self) -> str:
        return str(self.metadata.get("source_text", self.text[:200]))

    def to_source_reference(self) -> ChatSourceReference:
        return ChatSourceReference(
            chapter_id=self.chapter_id,
            chapter_number=self.chapter_number,
            chunk_id=self.id,
            memory_type=self.memory_type,
            relevant_excerpt=self.text[:350],
            similarity_score=self.similarity_score,
            distance=self.distance,
        )


class RetrievalService:
    """Service providing story-scoped, isolated vector retrieval from ChromaDB."""

    def __init__(
        self,
        chroma_manager: ChromaDBManager,
        embedding_service: EmbeddingService,
        story_repo: StoryRepository,
    ):
        self._chroma = chroma_manager
        self._embedding_service = embedding_service
        self._story_repo = story_repo

    def _deduplicate_chunks(
        self, chunks: List[RetrievedMemoryChunk], overlap_threshold: float = 0.80
    ) -> List[RetrievedMemoryChunk]:
        """
        Deduplicate chunks that have identical IDs or highly overlapping text.
        Preserves the highest-scoring chunk among near-duplicates.
        """
        deduped: List[RetrievedMemoryChunk] = []
        seen_ids = set()

        for chunk in chunks:
            if chunk.id in seen_ids:
                continue

            # Check overlap against already accepted chunks
            is_duplicate = False
            chunk_words = set(chunk.text.lower().split())
            if not chunk_words:
                continue

            for accepted in deduped:
                accepted_words = set(accepted.text.lower().split())
                if not accepted_words:
                    continue

                # Jaccard word similarity
                intersection = chunk_words.intersection(accepted_words)
                union = chunk_words.union(accepted_words)
                jaccard = len(intersection) / len(union) if union else 0.0

                if jaccard >= overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_ids.add(chunk.id)
                deduped.append(chunk)

        return deduped

    async def retrieve_relevant_memories(
        self,
        query: str,
        story_id: str,
        user_id: str,
        top_k: int = 5,
        min_relevance: float = 0.25,
        memory_type: Optional[str] = None,
    ) -> List[RetrievedMemoryChunk]:
        """
        Execute story-scoped semantic retrieval for an author's question.

        Enforces:
        - Strict user_id and story_id isolation.
        - Minimum relevance filtering (threshold).
        - Chunk deduplication.
        - Preserves source provenance.
        """
        if not query or not query.strip():
            raise ValidationException("Query string cannot be empty.")

        # 1. Enforce story existence and user ownership
        story_exists = await self._story_repo.exists(story_id=story_id, user_id=user_id)
        if not story_exists:
            raise StoryNotFoundException(story_id=story_id)

        # 2. Vectorize the query using Sentence Transformers
        query_vector = self._embedding_service.embed_text(query.strip(), normalize=True)

        # 3. Query ChromaDB with strict isolation filters
        # Request slightly more chunks (top_k * 2) to account for deduplication and thresholding
        fetch_k = min(top_k * 2, 40)
        raw_matches = self._chroma.search(
            query_embedding=query_vector,
            user_id=user_id,
            story_id=story_id,
            top_k=fetch_k,
            memory_type=memory_type,
        )

        if not raw_matches:
            logger.info(
                f"No memory matches found in ChromaDB for story '{story_id}' (query='{query[:50]}...')."
            )
            return []

        # 4. Filter by minimum relevance threshold and wrap in model
        candidates: List[RetrievedMemoryChunk] = []
        for m in raw_matches:
            score = float(m["similarity_score"])
            distance = float(m["distance"])
            if score >= min_relevance:
                candidates.append(
                    RetrievedMemoryChunk(
                        id=m["id"],
                        text=m["text"],
                        metadata=m["metadata"],
                        similarity_score=score,
                        distance=distance,
                    )
                )

        # 5. Sort candidates descending by similarity_score
        candidates.sort(key=lambda c: c.similarity_score, reverse=True)

        # 6. Deduplicate overlapping chunks
        unique_chunks = self._deduplicate_chunks(candidates)

        # 7. Take top_k
        final_chunks = unique_chunks[:top_k]

        logger.info(
            f"Retrieved {len(final_chunks)} relevant memory chunks for story '{story_id}' "
            f"(candidates={len(candidates)}, final top_k={top_k}, min_relevance={min_relevance})."
        )
        return final_chunks


def get_retrieval_service(
    chroma_manager: ChromaDBManager = Depends(get_chromadb_manager),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    story_repo: StoryRepository = Depends(get_story_repository),
) -> RetrievalService:
    """FastAPI dependency provider for RetrievalService."""
    return RetrievalService(
        chroma_manager=chroma_manager,
        embedding_service=embedding_service,
        story_repo=story_repo,
    )
