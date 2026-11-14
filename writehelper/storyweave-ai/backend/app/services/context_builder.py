"""
StoryWeave AI - RAG Context Builder Service

Constructs structured, grounded context from retrieved memory chunks while enforcing
strict character/token budgets, source provenance tagging, and explicit vs inferred distinctions.
Completely decoupled from Gemini.
"""

from typing import List, Dict, Any, Optional, Tuple
from app.services.retrieval_service import RetrievedMemoryChunk
from app.core.logging import get_logger

logger = get_logger("service.context_builder")


class BuiltContext:
    """Represents assembled grounded context and its inclusion metrics."""

    def __init__(
        self,
        formatted_context: str,
        included_chunks: List[RetrievedMemoryChunk],
        total_characters: int,
        budget_exhausted: bool,
    ):
        self.formatted_context = formatted_context
        self.included_chunks = included_chunks
        self.total_characters = total_characters
        self.budget_exhausted = budget_exhausted


class ContextBuilder:
    """Assembles retrieved memory chunks and story background into grounded prompt context."""

    def __init__(self, max_character_budget: int = 4000):
        self.max_character_budget = max_character_budget

    def build_context(
        self,
        chunks: List[RetrievedMemoryChunk],
        story_context: Optional[Dict[str, Any]] = None,
        budget_override: Optional[int] = None,
        character_context: Optional[List[Dict[str, Any]]] = None,
    ) -> BuiltContext:
        """
        Build structured context string from retrieved memories and characters within budget.

        Prioritizes highest-relevance memories and relevant character profiles.
        Maintains source provenance for every memory chunk.
        """
        budget = budget_override or self.max_character_budget
        char_section = self._build_character_section(character_context)

        if not chunks:
            # Build story background context and optional character section
            header = self._build_story_header(story_context)
            if char_section:
                body = f"{char_section}\n[NO MATCHING RAW MANUSCRIPT CHUNKS FOUND]\n"
            else:
                body = "\n[NO MATCHING STORY MEMORIES FOUND IN MANUSCRIPT REPOSITORY]\n"
            full_text = f"{header}\n{body}"
            return BuiltContext(
                formatted_context=full_text,
                included_chunks=[],
                total_characters=len(full_text),
                budget_exhausted=False,
            )

        header = self._build_story_header(story_context)
        current_len = len(header)
        
        # Include character section if fits within budget
        if char_section:
            if current_len + len(char_section) < budget - 200:
                header = f"{header}\n{char_section}"
                current_len = len(header)

        included_chunks: List[RetrievedMemoryChunk] = []
        memory_blocks: List[str] = []
        budget_exhausted = False

        # Sort descending by relevance
        sorted_chunks = sorted(chunks, key=lambda c: c.similarity_score, reverse=True)

        for idx, chunk in enumerate(sorted_chunks):
            block = (
                f"\n--- [Source Memory #{idx+1}] ---\n"
                f"Chapter: {chunk.chapter_number} (ID: {chunk.chapter_id})\n"
                f"Chunk ID: {chunk.id}\n"
                f"Memory Type: {chunk.memory_type}\n"
                f"Relevance Score: {chunk.similarity_score:.4f} (Cosine Similarity)\n"
                f"Manuscript Excerpt:\n\"\"\"{chunk.text.strip()}\"\"\"\n"
            )

            block_len = len(block)
            if current_len + block_len <= budget:
                memory_blocks.append(block)
                included_chunks.append(chunk)
                current_len += block_len
            else:
                # If budget is full and no chunks have been added yet, add a truncated version
                if not included_chunks:
                    allowed_text_len = max(100, budget - current_len - 250)
                    truncated_text = chunk.text.strip()[:allowed_text_len] + "... [truncated]"
                    fallback_block = (
                        f"\n--- [Source Memory #1 (Truncated)] ---\n"
                        f"Chapter: {chunk.chapter_number} (ID: {chunk.chapter_id})\n"
                        f"Chunk ID: {chunk.id}\n"
                        f"Memory Type: {chunk.memory_type}\n"
                        f"Manuscript Excerpt:\n\"\"\"{truncated_text}\"\"\"\n"
                    )
                    memory_blocks.append(fallback_block)
                    included_chunks.append(chunk)
                budget_exhausted = True
                break

        final_context = f"{header}\n{''.join(memory_blocks)}"
        logger.info(
            f"Built context with {len(included_chunks)}/{len(chunks)} chunks "
            f"({len(final_context)} chars, budget={budget}, exhausted={budget_exhausted})."
        )

        return BuiltContext(
            formatted_context=final_context,
            included_chunks=included_chunks,
            total_characters=len(final_context),
            budget_exhausted=budget_exhausted,
        )

    def _build_story_header(self, story_context: Optional[Dict[str, Any]]) -> str:
        """Construct the foundational story overview block."""
        if not story_context:
            return "### STORY UNIVERSE CONTEXT:"

        lines = ["### STORY UNIVERSE CONTEXT:"]
        title = story_context.get("title")
        if title:
            lines.append(f"Title: {title}")
        genre = story_context.get("genre")
        if genre:
            lines.append(f"Genre: {genre}")
        synopsis = story_context.get("synopsis")
        if synopsis:
            lines.append(f"Synopsis: {synopsis}")
        rules = story_context.get("world_rules_overview")
        if rules:
            lines.append(f"World Rules Overview: {rules}")

        return "\n".join(lines)


    def _build_character_section(self, character_context: Optional[List[Dict[str, Any]]]) -> str:
        """Construct structured summary of relevant characters."""
        if not character_context:
            return ""

        lines = ["\n### RELEVANT CHARACTER PROFILES:"]
        for c in character_context:
            name = c.get("name", "Unknown")
            aliases = ", ".join(c.get("aliases", []))
            alias_str = f" (Aliases: {aliases})" if aliases else ""
            lines.append(f"- Character: {name}{alias_str}")
            if c.get("physical_description"):
                lines.append(f"  Description: {c['physical_description']}")
            if c.get("personality_traits"):
                lines.append(f"  Personality Traits: {', '.join(c['personality_traits'])}")
            if c.get("abilities"):
                lines.append(f"  Abilities: {', '.join(c['abilities'])}")
            if c.get("motivations"):
                lines.append(f"  Motivations: {', '.join(c['motivations'])}")
            if c.get("goals"):
                lines.append(f"  Goals: {', '.join(c['goals'])}")
        return "\n".join(lines)


# Singleton instance
_context_builder: Optional[ContextBuilder] = None


def get_context_builder() -> ContextBuilder:
    """Dependency provider for ContextBuilder."""
    global _context_builder
    if _context_builder is None:
        _context_builder = ContextBuilder(max_character_budget=4000)
    return _context_builder
