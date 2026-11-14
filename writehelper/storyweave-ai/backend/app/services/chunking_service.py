"""
StoryWeave AI - Chapter Chunking Service

Divides manuscripts into meaningful semantic chunks along natural paragraph and
sentence boundaries while avoiding mid-sentence cuts.
Attaches source provenance metadata and assigns deterministic chunk IDs.
"""

import re
from typing import List, Dict, Any, Optional
from app.schemas.memory import MemoryChunk, MemoryType
from app.core.logging import get_logger

logger = get_logger("service.chunking")


class ChunkingService:
    """Service for splitting manuscripts into overlapping semantic chunks."""

    def __init__(
        self,
        target_chunk_size: int = 800,  # Characters (~120-160 words)
        chunk_overlap: int = 150,     # Characters (~25-35 words)
    ):
        if chunk_overlap >= target_chunk_size:
            raise ValueError("chunk_overlap must be strictly less than target_chunk_size.")
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences preserving punctuation."""
        # Normalize carriage returns
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        # Split on paragraph breaks or sentence endings (.!?) followed by whitespace
        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]

        sentences: List[str] = []
        for p in paragraphs:
            # Match sentence endings
            p_sentences = re.split(r"(?<=[.!?])\s+", p)
            for s in p_sentences:
                cleaned = s.strip()
                if cleaned:
                    sentences.append(cleaned)
        return sentences

    def chunk_chapter(
        self,
        chapter_id: str,
        story_id: str,
        user_id: str,
        chapter_number: int,
        content: str,
        memory_type: str = MemoryType.MANUSCRIPT_CHUNK.value,
    ) -> List[MemoryChunk]:
        """
        Split a chapter's manuscript into semantic chunks.

        Ensures:
        - Deterministic chunk IDs: f"{chapter_id}_chunk_{i}"
        - Sentence integrity (no cut-off sentences)
        - Overlap between consecutive chunks for narrative continuity
        - Full manuscript preservation (nothing silently discarded)
        - Rich source provenance attached to every chunk
        """
        if not content or not content.strip():
            return []

        raw_content = content.strip()
        sentences = self._split_into_sentences(raw_content)

        # Fallback if sentence splitting produced nothing (e.g. no punctuation)
        if not sentences:
            sentences = [raw_content]

        chunks_text: List[str] = []
        current_sentences: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If a single sentence is excessively long, handle by word splitting
            if sentence_len > self.target_chunk_size:
                if current_sentences:
                    chunks_text.append(" ".join(current_sentences))
                    current_sentences = []
                    current_len = 0

                # Break oversized sentence at word boundaries
                words = sentence.split()
                temp_chunk: List[str] = []
                temp_len = 0
                for w in words:
                    if temp_len + len(w) + 1 > self.target_chunk_size and temp_chunk:
                        chunks_text.append(" ".join(temp_chunk))
                        temp_chunk = []
                        temp_len = 0
                    temp_chunk.append(w)
                    temp_len += len(w) + 1
                if temp_chunk:
                    current_sentences.append(" ".join(temp_chunk))
                    current_len = temp_len
                continue

            # Standard accumulation
            if current_len + sentence_len + 1 > self.target_chunk_size and current_sentences:
                chunks_text.append(" ".join(current_sentences))

                # Build overlap from the end of current_sentences
                overlap_sentences: List[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) + 1 <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break

                current_sentences = overlap_sentences + [sentence]
                current_len = sum(len(s) + 1 for s in current_sentences)
            else:
                current_sentences.append(sentence)
                current_len += sentence_len + 1

        if current_sentences:
            final_text = " ".join(current_sentences)
            # Avoid emitting a pure duplicate of the last chunk due to overlap
            if not chunks_text or chunks_text[-1] != final_text:
                chunks_text.append(final_text)

        # Build final MemoryChunk objects with deterministic IDs
        memory_chunks: List[MemoryChunk] = []
        for idx, text in enumerate(chunks_text):
            chunk_id = f"{chapter_id}_chunk_{idx}"
            metadata = {
                "user_id": str(user_id),
                "story_id": str(story_id),
                "chapter_id": str(chapter_id),
                "chapter_number": int(chapter_number),
                "chunk_index": int(idx),
                "memory_type": str(memory_type),
                "word_count": int(len(text.split())),
                "char_count": int(len(text)),
                "source_text": text[:300],  # Provenance preview
            }
            memory_chunks.append(
                MemoryChunk(
                    id=chunk_id,
                    text=text,
                    metadata=metadata,
                )
            )

        logger.info(
            f"Chunked chapter {chapter_number} ({len(raw_content)} chars) into "
            f"{len(memory_chunks)} semantic chunks (ID prefix: {chapter_id}_chunk_)."
        )
        return memory_chunks


# Default singleton instance
_chunking_service: Optional[ChunkingService] = None


def get_chunking_service() -> ChunkingService:
    """Dependency provider for ChunkingService."""
    global _chunking_service
    if _chunking_service is None:
        _chunking_service = ChunkingService()
    return _chunking_service
