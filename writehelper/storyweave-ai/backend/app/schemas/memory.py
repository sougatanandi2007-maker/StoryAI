"""
StoryWeave AI - Memory Schemas

Pydantic validation schemas for vector memory chunking, indexing, and semantic search.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from app.schemas.common import PaginatedResponse


class MemoryType(str, Enum):
    """Types of memory chunks stored in ChromaDB."""

    MANUSCRIPT_CHUNK = "manuscript_chunk"
    CHARACTER_MEMORY = "character_memory"
    EVENT_MEMORY = "event_memory"
    WORLD_RULE_MEMORY = "world_rule_memory"
    FACT_MEMORY = "fact_memory"


class MemoryChunkMetadata(BaseModel):
    """Source provenance and categorization metadata for a memory chunk."""

    user_id: str
    story_id: str
    chapter_id: str
    chapter_number: int
    chunk_index: int
    memory_type: str = MemoryType.MANUSCRIPT_CHUNK.value
    word_count: int
    source_text: str


class MemoryChunk(BaseModel):
    """A single semantic text chunk ready for vector embedding and storage."""

    id: str = Field(..., description="Deterministic chunk ID (e.g. chap_xxx_chunk_0)")
    text: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(..., description="ChromaDB-compatible flat metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Dense vector embedding")


class MemorySearchRequest(BaseModel):
    """Payload for executing a semantic search across story universe memories."""

    query: str = Field(..., min_length=1, max_length=1000, description="Natural language search query")
    story_id: str = Field(..., min_length=1, description="Target story universe ID")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of memory results to return")
    memory_type: Optional[str] = Field(
        default=None,
        description="Filter by memory type (e.g. 'manuscript_chunk')",
    )


class MemorySearchResult(BaseModel):
    """Individual retrieved memory match."""

    id: str = Field(..., description="Unique chunk ID")
    text: str = Field(..., description="Stored manuscript memory text")
    metadata: Dict[str, Any] = Field(..., description="Source provenance metadata")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    distance: float = Field(..., description="Cosine distance metric from ChromaDB")


class MemorySearchResponse(BaseModel):
    """Search results response with provenance and score metrics."""

    query: str
    story_id: str
    total_found: int
    results: List[MemorySearchResult]


class StoredMemoryItem(BaseModel):
    """Representation of an existing memory chunk in ChromaDB."""

    id: str
    text: str
    metadata: Dict[str, Any]


class PaginatedMemoryResponse(PaginatedResponse[StoredMemoryItem]):
    """Paginated list of stored memories for a story."""
    pass


class IndexingReport(BaseModel):
    """Status report returned after indexing a chapter's manuscript into ChromaDB."""

    chapter_id: str
    story_id: str
    chunks_indexed: int
    word_count: int
    status: str = "completed"
    message: str


class StoryMemoryStatusResponse(BaseModel):
    """Memory coverage and chunk statistics for a story universe."""

    story_id: str
    total_chunks: int
    indexed_chapters_count: int
    indexed_chapter_ids: List[str]
    is_indexed: bool
