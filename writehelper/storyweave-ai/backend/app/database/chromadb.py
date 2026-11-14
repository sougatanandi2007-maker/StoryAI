"""
StoryWeave AI - ChromaDB Connection Manager

Manages the persistent local ChromaDB client for semantic story universe memories.
Configures cosine distance space, deterministic chunk upserts, story/user isolation,
and granular metadata querying.
"""

from typing import Optional, List, Dict, Any, Tuple
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import MemoryServiceError

logger = get_logger("database.chromadb")


class ChromaDBManager:
    """Manages ChromaDB vector store client lifecycle and memory operations."""

    def __init__(self, persist_directory: Optional[str] = None):
        self._custom_persist_directory = persist_directory
        self._client = None
        self._collection = None

    def connect(self) -> None:
        """Initialize the persistent ChromaDB client with cosine distance space."""
        settings = get_settings()
        path = self._custom_persist_directory or settings.chroma_persist_directory

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            # Ensure collection configured with cosine distance
            self._collection = self._client.get_or_create_collection(
                name="story_memories",
                metadata={"hnsw:space": "cosine", "description": "Persistent story universe memories"},
            )

            count = self._collection.count()
            logger.info(
                f"ChromaDB connected at '{path}' — collection 'story_memories' ({count} memories)."
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB at '{path}': {e}")
            self._client = None
            self._collection = None
            raise MemoryServiceError(f"Failed to connect to ChromaDB: {str(e)}")

    def disconnect(self) -> None:
        """Clean up ChromaDB resources."""
        self._client = None
        self._collection = None
        logger.info("ChromaDB disconnected.")

    @property
    def client(self):
        return self._client

    @property
    def collection(self):
        return self._collection

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._collection is not None

    def _ensure_connected(self) -> None:
        """Ensure active connection or connect on demand."""
        if not self.is_connected:
            self.connect()

    def upsert_chunks(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """
        Idempotently insert or update memory chunks in ChromaDB.

        Deterministic chunk IDs ensure reprocessing updates existing vectors
        without creating duplicate chunks.
        """
        self._ensure_connected()

        if not ids:
            return 0

        try:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(f"Upserted {len(ids)} memory chunks into ChromaDB.")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to upsert chunks into ChromaDB: {e}")
            raise MemoryServiceError(f"ChromaDB upsert failed: {str(e)}")

    def search(
        self,
        query_embedding: List[float],
        user_id: str,
        story_id: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search with strict story and user isolation filters.

        Uses Cosine distance. Returns matches with similarity_score = 1.0 - distance.
        """
        self._ensure_connected()

        # Build isolated filter
        filter_conditions = [
            {"user_id": {"$eq": str(user_id)}},
            {"story_id": {"$eq": str(story_id)}},
        ]
        if memory_type:
            filter_conditions.append({"memory_type": {"$eq": str(memory_type)}})

        where_filter = {"$and": filter_conditions}

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            matches: List[Dict[str, Any]] = []
            if not results or not results["ids"] or not results["ids"][0]:
                return matches

            ids = results["ids"][0]
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            space = "l2"
            if self._collection.metadata and isinstance(self._collection.metadata, dict):
                space = self._collection.metadata.get("hnsw:space", "l2")

            for i in range(len(ids)):
                dist = float(distances[i])
                if space == "cosine":
                    similarity = max(0.0, min(1.0, 1.0 - dist))
                elif space == "ip":
                    similarity = max(0.0, min(1.0, dist))
                else:
                    # L2 distance on unit-normalized vectors: ||u - v||^2 = 2 - 2*cos => cos = 1 - dist/2
                    similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

                matches.append(
                    {
                        "id": ids[i],
                        "text": docs[i],
                        "metadata": metadatas[i],
                        "distance": dist,
                        "similarity_score": round(similarity, 4),
                    }
                )

            return matches
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            raise MemoryServiceError(f"ChromaDB search query failed: {str(e)}")

    def delete_by_chapter(self, chapter_id: str, user_id: str) -> None:
        """Delete all memory chunks for a chapter with user isolation."""
        self._ensure_connected()
        where_filter = {
            "$and": [
                {"chapter_id": {"$eq": str(chapter_id)}},
                {"user_id": {"$eq": str(user_id)}},
            ]
        }
        try:
            self._collection.delete(where=where_filter)
            logger.info(f"Deleted ChromaDB memories for chapter '{chapter_id}'.")
        except Exception as e:
            logger.error(f"Failed to delete chapter memories: {e}")
            raise MemoryServiceError(f"ChromaDB delete by chapter failed: {str(e)}")

    def delete_by_story(self, story_id: str, user_id: str) -> None:
        """Delete all memory chunks for a story with user isolation."""
        self._ensure_connected()
        where_filter = {
            "$and": [
                {"story_id": {"$eq": str(story_id)}},
                {"user_id": {"$eq": str(user_id)}},
            ]
        }
        try:
            self._collection.delete(where=where_filter)
            logger.info(f"Deleted ChromaDB memories for story '{story_id}'.")
        except Exception as e:
            logger.error(f"Failed to delete story memories: {e}")
            raise MemoryServiceError(f"ChromaDB delete by story failed: {str(e)}")

    def get_memories_for_story(
        self,
        story_id: str,
        user_id: str,
        chapter_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Retrieve stored memory records for a story with pagination."""
        self._ensure_connected()

        filter_conditions = [
            {"story_id": {"$eq": str(story_id)}},
            {"user_id": {"$eq": str(user_id)}},
        ]
        if chapter_id:
            filter_conditions.append({"chapter_id": {"$eq": str(chapter_id)}})

        where_filter = {"$and": filter_conditions}

        try:
            # Fetch matching records
            results = self._collection.get(
                where=where_filter,
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"],
            )

            # Total count for this filter
            all_records = self._collection.get(
                where=where_filter,
                include=[],
            )
            total = len(all_records["ids"]) if all_records and "ids" in all_records else 0

            items: List[Dict[str, Any]] = []
            if results and results.get("ids"):
                for i in range(len(results["ids"])):
                    items.append(
                        {
                            "id": results["ids"][i],
                            "text": results["documents"][i],
                            "metadata": results["metadatas"][i],
                        }
                    )
            return items, total
        except Exception as e:
            logger.error(f"Failed to get story memories: {e}")
            raise MemoryServiceError(f"ChromaDB get memories failed: {str(e)}")

    def health_check(self) -> dict:
        """Return detailed health status for ChromaDB."""
        if self._client is None or self._collection is None:
            settings = get_settings()
            path = self._custom_persist_directory or settings.chroma_persist_directory
            return {
                "status": "disconnected",
                "message": "ChromaDB client is not initialized",
                "persist_directory": path,
            }

        try:
            count = self._collection.count()
            settings = get_settings()
            path = self._custom_persist_directory or settings.chroma_persist_directory
            return {
                "status": "healthy",
                "message": "ChromaDB is operational",
                "collection": "story_memories",
                "document_count": count,
                "persist_directory": path,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"ChromaDB health check failed: {str(e)}",
            }


# Default singleton instance
chromadb_manager = ChromaDBManager()


def get_chromadb_manager() -> ChromaDBManager:
    """Dependency provider for ChromaDBManager."""
    return chromadb_manager
