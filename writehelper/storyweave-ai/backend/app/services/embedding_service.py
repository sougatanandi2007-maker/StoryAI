"""
StoryWeave AI - Embedding Service

Generates dense vector embeddings using Sentence Transformers (all-MiniLM-L6-v2).
Implements thread-safe singleton loading, batch processing, normalization,
and input validation.
"""

import threading
from typing import List, Optional, Any
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import EmbeddingServiceError

logger = get_logger("service.embedding")


class EmbeddingService:
    """Service for loading SentenceTransformer models and generating vector embeddings."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self._model_name = model_name or settings.embedding_model
        self._model: Optional[Any] = None
        self._lock = threading.Lock()
        self._dimension: int = 384  # Default for all-MiniLM-L6-v2

    def _load_model(self) -> Any:
        """Load the SentenceTransformer model once with thread safety."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading SentenceTransformer embedding model '{self._model_name}'...")
                # First try loading from local cache to avoid hanging on network in offline/sandboxed environments
                try:
                    self._model = SentenceTransformer(self._model_name, local_files_only=True)
                except Exception:
                    self._model = SentenceTransformer(self._model_name)

                # Verify dimension
                test_vec = self._model.encode("test", normalize_embeddings=True)
                self._dimension = len(test_vec)
                logger.info(
                    f"Successfully loaded model '{self._model_name}' (dimension={self._dimension})."
                )
                return self._model
            except Exception as e:
                logger.error(f"Failed to load embedding model '{self._model_name}': {e}")
                raise EmbeddingServiceError(
                    f"Failed to load embedding model '{self._model_name}': {str(e)}"
                )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def embed_text(self, text: str, normalize: bool = True) -> List[float]:
        """
        Generate a normalized vector embedding for a single text string.

        Raises ValueError on empty or whitespace-only text.
        """
        if not text or not text.strip():
            raise ValueError("Input text for embedding cannot be empty.")

        model = self._load_model()
        try:
            vector = model.encode(text.strip(), normalize_embeddings=normalize)
            return vector.tolist() if hasattr(vector, "tolist") else list(vector)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingServiceError(f"Failed to generate text embedding: {str(e)}")

    def embed_batch(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """
        Generate normalized vector embeddings for a batch of text strings.

        Raises ValueError on empty list or invalid items.
        """
        if not texts:
            raise ValueError("Texts batch cannot be empty.")

        cleaned_texts: List[str] = []
        for idx, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(f"Text at index {idx} is empty or whitespace only.")
            cleaned_texts.append(t.strip())

        model = self._load_model()
        try:
            vectors = model.encode(
                cleaned_texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=normalize,
            )
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise EmbeddingServiceError(f"Failed to generate batch embeddings: {str(e)}")


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Dependency provider for EmbeddingService."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
