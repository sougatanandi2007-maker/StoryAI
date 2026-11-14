"""
StoryWeave AI - Chapter Summarization Service

Local, thread-safe chapter summarization using Hugging Face's
DistilBART model ("sshleifer/distilbart-cnn-12-6").
Preserves character names, chronological events, and core narrative developments.
Supports recursive chunking for long manuscripts exceeding the model input limits.
"""

import math
import re
import threading
from typing import List, Optional, Any, Tuple
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import ValidationException

logger = get_logger("service.summarization")


class SummarizationServiceError(Exception):
    """Exception raised when local summarization fails."""
    pass


class SummarizationService:
    """Service for local DistilBART chapter summarization with thread-safe singleton loading."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self._model_name = model_name or settings.summarization_model
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._lock = threading.Lock()
        self._max_chunk_words: int = settings.summarization_max_chunk_words
        self._overlap_words: int = settings.summarization_overlap_words

    def _load_model(self) -> Tuple[Any, Any]:
        """Lazy thread-safe loading of the DistilBART model and tokenizer.

        Loaded once and reused for all subsequent chapter summarizations.
        """
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        with self._lock:
            if self._tokenizer is not None and self._model is not None:
                return self._tokenizer, self._model

            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

                logger.info(f"Loading local summarization model '{self._model_name}'...")
                # First attempt local cache only
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, local_files_only=True)
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name, local_files_only=True)
                except Exception:
                    self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)

                # Ensure model is in eval mode on CPU
                self._model.eval()
                logger.info(f"Successfully initialized DistilBART summarization model '{self._model_name}'.")
                return self._tokenizer, self._model
            except Exception as e:
                logger.error(f"Failed to load DistilBART model '{self._model_name}': {e}")
                raise SummarizationServiceError(f"Failed to load summarization model '{self._model_name}': {str(e)}")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def chunk_text(self, text: str, max_chunk_words: Optional[int] = None, overlap_words: Optional[int] = None) -> List[str]:
        """
        Split a chapter manuscript into sentence-boundary preserved chunks.
        Ensures each chunk safely fits within DistilBART's 1024 token limit.
        """
        max_words = max_chunk_words or self._max_chunk_words
        overlap = overlap_words or self._overlap_words

        if not text or not text.strip():
            return []

        # Split on natural sentence boundaries
        raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences:
            return [text.strip()]

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_word_count = 0

        for sentence in sentences:
            s_words = sentence.split()
            s_count = len(s_words)
            if not s_words:
                continue

            current_sentences.append(sentence)
            current_word_count += s_count

            if current_word_count >= max_words:
                chunks.append(" ".join(current_sentences).strip())

                # Compute sentence overlap
                overlap_count = 0
                overlap_sentences: List[str] = []
                for s in reversed(current_sentences):
                    overlap_count += len(s.split())
                    overlap_sentences.insert(0, s)
                    if overlap_count >= overlap:
                        break

                current_sentences = overlap_sentences
                current_word_count = sum(len(s.split()) for s in current_sentences)

        if current_sentences:
            remainder_text = " ".join(current_sentences).strip()
            # Only append if not identical to the last chunk
            if not chunks or remainder_text != chunks[-1]:
                chunks.append(remainder_text)

        return chunks

    def _summarize_single_passage(
        self,
        passage: str,
        max_summary_length: int = 142,
        min_summary_length: int = 40,
    ) -> str:
        """Execute a single forward-pass summarization with DistilBART."""
        tokenizer, model = self._load_model()

        inputs = tokenizer(
            passage,
            max_length=1024,
            return_tensors="pt",
            truncation=True,
        )

        # Compute dynamic max length based on input size
        input_token_len = inputs["input_ids"].shape[1]
        effective_max = min(max_summary_length, max(30, int(input_token_len * 0.75)))
        effective_min = min(min_summary_length, max(15, int(effective_max * 0.35)))

        import torch
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=effective_max,
                min_length=effective_min,
                num_beams=2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        # Clean double spaces or artifacts
        return re.sub(r'\s+', ' ', summary_text)

    def summarize_chapter(
        self,
        content: str,
        max_summary_length: int = 160,
        min_summary_length: int = 40,
    ) -> str:
        """
        Generate a concise, faithful chapter summary using DistilBART.

        Preserves:
        - Character names and identities
        - Chronological order of key events
        - Key relationships and revelations
        - Strictly grounded in submitted chapter content

        Long manuscripts are processed using multi-chunk hierarchical summarization.
        """
        if not content or not content.strip():
            raise ValidationException("Chapter content is empty; cannot generate summary.")

        cleaned_text = content.strip()
        words = cleaned_text.split()

        # 1. Short Chapter (single chunk under limit)
        if len(words) <= self._max_chunk_words:
            try:
                return self._summarize_single_passage(
                    cleaned_text,
                    max_summary_length=max_summary_length,
                    min_summary_length=min_summary_length,
                )
            except Exception as e:
                logger.error(f"DistilBART single-passage summarization error: {e}")
                raise SummarizationServiceError(f"Summarization failed: {str(e)}")

        # 2. Long Chapter (multi-chunk hierarchical summarization)
        chunks = self.chunk_text(cleaned_text)
        if not chunks:
            chunks = [cleaned_text]

        chunk_summaries: List[str] = []
        for idx, chunk in enumerate(chunks):
            try:
                c_sum = self._summarize_single_passage(
                    chunk,
                    max_summary_length=110,
                    min_summary_length=25,
                )
                if c_sum:
                    chunk_summaries.append(c_sum)
            except Exception as e:
                logger.warning(f"Failed to summarize chunk {idx+1}/{len(chunks)}: {e}")
                # Continue processing remaining chunks to preserve overall narrative
                continue

        if not chunk_summaries:
            raise SummarizationServiceError("Failed to generate summaries for chapter chunks.")

        # If only one chunk summary produced, return it
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        # Combine chunk summaries in chronological order
        combined_draft = " ".join(chunk_summaries)
        
        # If combined text is concise enough, return directly
        if len(combined_draft.split()) <= max_summary_length:
            return combined_draft

        # Second-stage consolidation pass to produce a unified chapter summary
        try:
            final_summary = self._summarize_single_passage(
                combined_draft,
                max_summary_length=max_summary_length,
                min_summary_length=min_summary_length,
            )
            return final_summary
        except Exception as e:
            logger.warning(f"Consolidation pass failed, falling back to chronological chunk summaries: {e}")
            return combined_draft


# Singleton instance
_summarization_service: Optional[SummarizationService] = None


def get_summarization_service() -> SummarizationService:
    """Dependency provider for SummarizationService."""
    global _summarization_service
    if _summarization_service is None:
        _summarization_service = SummarizationService()
    return _summarization_service

