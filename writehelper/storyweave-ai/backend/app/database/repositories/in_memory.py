"""
StoryWeave AI - In-Memory Repository Implementation

=============================================================================
WARNING: TEMPORARY IN-MEMORY STORAGE
This storage layer runs entirely in server memory.
Data is NOT persistent and WILL BE LOST whenever the application restarts.
Used for Phase 2 development and testing until MongoDB Atlas is configured.
=============================================================================
"""

import copy
import threading
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict, Any
from app.database.repositories.base import (
    StoryRepository,
    ChapterRepository,
    AnalysisRepository,
    ContinuityRepository,
    CharacterRepository,
)
from app.core.logging import get_logger

logger = get_logger("repository.in_memory")


class InMemoryStoryRepository(StoryRepository):
    """
    Thread-safe in-memory implementation of StoryRepository.

    Stores story documents in a Python dictionary.
    NOTE: Data will be reset on server restart.
    """

    def __init__(self):
        self._stories: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        logger.info(
            "Initialized InMemoryStoryRepository (TEMPORARY: non-persistent, resets on restart)"
        )

    async def create(self, story_data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            doc = copy.deepcopy(story_data)
            self._stories[doc["id"]] = doc
            return copy.deepcopy(doc)

    async def get_by_id(self, story_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            story = self._stories.get(story_id)
            if story and story.get("user_id") == user_id:
                return copy.deepcopy(story)
            return None

    async def list(self, user_id: str, skip: int = 0, limit: int = 20) -> Tuple[List[Dict[str, Any]], int]:
        with self._lock:
            user_stories = [
                copy.deepcopy(s)
                for s in self._stories.values()
                if s.get("user_id") == user_id
            ]
            # Sort newest first
            user_stories.sort(
                key=lambda x: x.get("created_at", datetime.min.replace(tzinfo=timezone.utc)),
                reverse=True,
            )
            total = len(user_stories)
            paginated = user_stories[skip : skip + limit]
            return paginated, total

    async def update(self, story_id: str, user_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            story = self._stories.get(story_id)
            if not story or story.get("user_id") != user_id:
                return None

            for key, val in update_data.items():
                if val is not None and key not in ("id", "user_id", "created_at"):
                    story[key] = copy.deepcopy(val)

            story["updated_at"] = datetime.now(timezone.utc)
            return copy.deepcopy(story)

    async def delete(self, story_id: str, user_id: str) -> bool:
        with self._lock:
            story = self._stories.get(story_id)
            if story and story.get("user_id") == user_id:
                del self._stories[story_id]
                return True
            return False

    async def exists(self, story_id: str, user_id: str) -> bool:
        with self._lock:
            story = self._stories.get(story_id)
            return bool(story and story.get("user_id") == user_id)

    def clear(self) -> None:
        """Clear all storage (for testing)."""
        with self._lock:
            self._stories.clear()


class InMemoryChapterRepository(ChapterRepository):
    """
    Thread-safe in-memory implementation of ChapterRepository.

    Guarantees preservation of the original manuscript (`original_content`)
    even when subsequent updates edit the working draft (`content`).
    NOTE: Data will be reset on server restart.
    """

    def __init__(self):
        self._chapters: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        logger.info(
            "Initialized InMemoryChapterRepository (TEMPORARY: non-persistent, resets on restart)"
        )

    async def create(self, chapter_data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            doc = copy.deepcopy(chapter_data)
            # Ensure immutable preservation of the original manuscript
            if "original_content" not in doc:
                doc["original_content"] = doc["content"]
            self._chapters[doc["id"]] = doc
            return copy.deepcopy(doc)

    async def get_by_id(self, chapter_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            ch = self._chapters.get(chapter_id)
            if ch and ch.get("user_id") == user_id:
                return copy.deepcopy(ch)
            return None

    async def list_by_story(
        self, story_id: str, user_id: str, skip: int = 0, limit: int = 20
    ) -> Tuple[List[Dict[str, Any]], int]:
        with self._lock:
            story_chapters = [
                copy.deepcopy(ch)
                for ch in self._chapters.values()
                if ch.get("story_id") == story_id and ch.get("user_id") == user_id
            ]
            # Sort by chapter sequence number ascending
            story_chapters.sort(key=lambda x: x.get("chapter_number", 0))
            total = len(story_chapters)
            paginated = story_chapters[skip : skip + limit]
            return paginated, total

    async def count_by_story(self, story_id: str, user_id: str) -> int:
        with self._lock:
            return sum(
                1
                for ch in self._chapters.values()
                if ch.get("story_id") == story_id and ch.get("user_id") == user_id
            )

    async def get_by_number(self, story_id: str, chapter_number: int, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for ch in self._chapters.values():
                if (
                    ch.get("story_id") == story_id
                    and ch.get("chapter_number") == chapter_number
                    and ch.get("user_id") == user_id
                ):
                    return copy.deepcopy(ch)
            return None

    async def update(self, chapter_id: str, user_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            ch = self._chapters.get(chapter_id)
            if not ch or ch.get("user_id") != user_id:
                return None

            for key, val in update_data.items():
                # PROTECT original_content: NEVER overwrite original manuscript!
                if key in ("id", "story_id", "user_id", "created_at", "original_content"):
                    continue
                ch[key] = copy.deepcopy(val) if val is not None else None

            # Recompute word count if content was updated
            if "content" in update_data and update_data["content"] is not None:
                ch["word_count"] = len(update_data["content"].split())

            ch["updated_at"] = datetime.now(timezone.utc)
            return copy.deepcopy(ch)

    async def delete(self, chapter_id: str, user_id: str) -> bool:
        with self._lock:
            ch = self._chapters.get(chapter_id)
            if ch and ch.get("user_id") == user_id:
                del self._chapters[chapter_id]
                return True
            return False

    async def delete_by_story(self, story_id: str, user_id: str) -> int:
        with self._lock:
            to_delete = [
                cid
                for cid, ch in self._chapters.items()
                if ch.get("story_id") == story_id and ch.get("user_id") == user_id
            ]
            for cid in to_delete:
                del self._chapters[cid]
            return len(to_delete)

    def clear(self) -> None:
        """Clear all storage (for testing)."""
        with self._lock:
            self._chapters.clear()


class InMemoryAnalysisRepository(AnalysisRepository):
    """
    Thread-safe in-memory implementation of AnalysisRepository.

    Stores chapter analysis reports and ensures idempotency:
    re-running analysis on a chapter updates the existing report
    rather than creating duplicate documents.
    NOTE: Data will be reset on server restart.
    """

    def __init__(self):
        self._reports_by_id: Dict[str, Dict[str, Any]] = {}
        self._reports_by_chapter: Dict[str, str] = {}  # chapter_id -> report_id
        self._lock = threading.RLock()
        logger.info(
            "Initialized InMemoryAnalysisRepository (TEMPORARY: non-persistent, resets on restart)"
        )

    async def save_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            doc = copy.deepcopy(report_data)
            chapter_id = doc["chapter_id"]

            # Check if report already exists for this chapter (idempotent upsert)
            existing_id = self._reports_by_chapter.get(chapter_id)
            if existing_id and existing_id in self._reports_by_id:
                # Update existing report
                existing = self._reports_by_id[existing_id]
                existing["analysis"] = doc["analysis"]
                existing["model_used"] = doc["model_used"]
                existing["updated_at"] = doc.get("updated_at", datetime.now(timezone.utc))
                return copy.deepcopy(existing)

            # Store new report
            self._reports_by_id[doc["id"]] = doc
            self._reports_by_chapter[chapter_id] = doc["id"]
            return copy.deepcopy(doc)

    async def get_by_chapter_id(self, chapter_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            report_id = self._reports_by_chapter.get(chapter_id)
            if not report_id:
                return None
            report = self._reports_by_id.get(report_id)
            if report and report.get("user_id") == user_id:
                return copy.deepcopy(report)
            return None

    async def delete_by_chapter_id(self, chapter_id: str, user_id: str) -> bool:
        with self._lock:
            report_id = self._reports_by_chapter.get(chapter_id)
            if not report_id:
                return False
            report = self._reports_by_id.get(report_id)
            if report and report.get("user_id") == user_id:
                del self._reports_by_id[report_id]
                del self._reports_by_chapter[chapter_id]
                return True
            return False

    async def delete_by_story_id(self, story_id: str, user_id: str) -> int:
        with self._lock:
            to_delete = [
                rid
                for rid, rep in self._reports_by_id.items()
                if rep.get("story_id") == story_id and rep.get("user_id") == user_id
            ]
            for rid in to_delete:
                rep = self._reports_by_id.pop(rid, None)
                if rep:
                    self._reports_by_chapter.pop(rep["chapter_id"], None)
            return len(to_delete)

    def clear(self) -> None:
        """Clear all storage (for testing)."""
        with self._lock:
            self._reports_by_id.clear()
            self._reports_by_chapter.clear()


class InMemoryContinuityRepository(ContinuityRepository):
    """
    Thread-safe in-memory implementation of ContinuityRepository.

    Stores chapter continuity reports and tracks author decisions on detected issues.
    Ensures idempotency and preserves author status decisions on repeated checks.
    NOTE: Data will be reset on server restart.
    """

    def __init__(self):
        self._reports_by_id: Dict[str, Dict[str, Any]] = {}
        self._reports_by_chapter: Dict[str, str] = {}  # chapter_id -> report_id
        self._lock = threading.RLock()
        logger.info(
            "Initialized InMemoryContinuityRepository (TEMPORARY: non-persistent, resets on restart)"
        )

    async def save_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            doc = copy.deepcopy(report_data)
            chapter_id = doc["chapter_id"]
            report_id = doc.get("report_id") or doc.get("id")
            doc["id"] = report_id
            doc["report_id"] = report_id

            existing_id = self._reports_by_chapter.get(chapter_id)
            if existing_id and existing_id in self._reports_by_id:
                existing = self._reports_by_id[existing_id]

                # Map existing issue statuses to preserve author triage choices
                status_map = {}
                notes_map = {}
                for old_issue in existing.get("issues", []):
                    key = (old_issue.get("issue_type"), old_issue.get("existing_fact"), old_issue.get("new_fact"))
                    status_map[key] = old_issue.get("status")
                    notes_map[key] = old_issue.get("author_notes")
                    # Also map by old issue_id directly
                    if "issue_id" in old_issue:
                        status_map[old_issue["issue_id"]] = old_issue.get("status")
                        notes_map[old_issue["issue_id"]] = old_issue.get("author_notes")

                # Apply preserved statuses to incoming issues if matching
                for new_issue in doc.get("issues", []):
                    key = (new_issue.get("issue_type"), new_issue.get("existing_fact"), new_issue.get("new_fact"))
                    issue_id = new_issue.get("issue_id")
                    if key in status_map and status_map[key] != "open":
                        new_issue["status"] = status_map[key]
                        new_issue["author_notes"] = notes_map.get(key)
                    elif issue_id in status_map and status_map[issue_id] != "open":
                        new_issue["status"] = status_map[issue_id]
                        new_issue["author_notes"] = notes_map.get(issue_id)

                existing["total_issues"] = len(doc.get("issues", []))
                existing["issues"] = doc.get("issues", [])
                existing["summary"] = doc.get("summary", existing.get("summary"))
                existing["checked_at"] = doc.get("checked_at", datetime.now(timezone.utc))
                return copy.deepcopy(existing)

            # Store fresh report
            self._reports_by_id[report_id] = doc
            self._reports_by_chapter[chapter_id] = report_id
            return copy.deepcopy(doc)

    async def get_by_id(self, report_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            report = self._reports_by_id.get(report_id)
            if report and report.get("user_id") == user_id:
                return copy.deepcopy(report)
            return None

    async def get_by_chapter_id(self, chapter_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            report_id = self._reports_by_chapter.get(chapter_id)
            if not report_id:
                return None
            report = self._reports_by_id.get(report_id)
            if report and report.get("user_id") == user_id:
                return copy.deepcopy(report)
            return None

    async def list_by_story(self, story_id: str, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            reports = [
                copy.deepcopy(rep)
                for rep in self._reports_by_id.values()
                if rep.get("story_id") == story_id and rep.get("user_id") == user_id
            ]
            reports.sort(key=lambda x: x.get("chapter_number", 0))
            return reports

    async def update_issue_status(
        self, issue_id: str, user_id: str, status: str, author_notes: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            for rep in self._reports_by_id.values():
                if rep.get("user_id") != user_id:
                    continue
                for issue in rep.get("issues", []):
                    if issue.get("issue_id") == issue_id:
                        issue["status"] = status
                        if author_notes is not None:
                            issue["author_notes"] = author_notes
                        issue["updated_at"] = datetime.now(timezone.utc)
                        return copy.deepcopy(issue)
            return None

    async def delete_by_chapter_id(self, chapter_id: str, user_id: str) -> bool:
        with self._lock:
            report_id = self._reports_by_chapter.get(chapter_id)
            if not report_id:
                return False
            report = self._reports_by_id.get(report_id)
            if report and report.get("user_id") == user_id:
                del self._reports_by_id[report_id]
                del self._reports_by_chapter[chapter_id]
                return True
            return False

    async def delete_by_story_id(self, story_id: str, user_id: str) -> int:
        with self._lock:
            to_delete = [
                rid
                for rid, rep in self._reports_by_id.items()
                if rep.get("story_id") == story_id and rep.get("user_id") == user_id
            ]
            for rid in to_delete:
                rep = self._reports_by_id.pop(rid, None)
                if rep:
                    self._reports_by_chapter.pop(rep["chapter_id"], None)
            return len(to_delete)

    def clear(self) -> None:
        """Clear all storage (for testing)."""
        with self._lock:
            self._reports_by_id.clear()
            self._reports_by_chapter.clear()


class InMemoryCharacterRepository(CharacterRepository):
    """
    Thread-safe in-memory implementation of CharacterRepository.

    Stores character profiles and relationships, with deduplication by name
    within a story scope. Supports case-insensitive exact matching on name
    and aliases.
    NOTE: Data will be reset on server restart.
    """

    def __init__(self):
        self._characters: Dict[str, Dict[str, Any]] = {}  # character_id -> character
        self._relationships: Dict[str, Dict[str, Any]] = {}  # relationship_id -> relationship
        self._lock = threading.RLock()
        logger.info(
            "Initialized InMemoryCharacterRepository (TEMPORARY: non-persistent, resets on restart)"
        )

    async def save_character(self, char_data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            doc = copy.deepcopy(char_data)
            story_id = doc["story_id"]
            user_id = doc["user_id"]
            name_lower = doc["name"].lower()

            # Deduplicate: if a character with the same name exists in the same
            # story for the same user, return the existing one instead.
            for existing in self._characters.values():
                if (
                    existing.get("story_id") == story_id
                    and existing.get("user_id") == user_id
                    and existing.get("name", "").lower() == name_lower
                ):
                    return copy.deepcopy(existing)

            self._characters[doc["character_id"]] = doc
            return copy.deepcopy(doc)

    async def get_by_id(self, character_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            char = self._characters.get(character_id)
            if char and char.get("user_id") == user_id:
                return copy.deepcopy(char)
            return None

    async def list_by_story(self, story_id: str, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                copy.deepcopy(ch)
                for ch in self._characters.values()
                if ch.get("story_id") == story_id and ch.get("user_id") == user_id
            ]

    async def find_by_name(
        self, story_id: str, user_id: str, name: str
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            name_lower = name.lower()
            for ch in self._characters.values():
                if (
                    ch.get("story_id") == story_id
                    and ch.get("user_id") == user_id
                    and ch.get("name", "").lower() == name_lower
                ):
                    return copy.deepcopy(ch)
            return None

    async def find_by_name_or_alias(
        self, story_id: str, user_id: str, name: str
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            name_lower = name.lower()
            for ch in self._characters.values():
                if ch.get("story_id") != story_id or ch.get("user_id") != user_id:
                    continue
                # Exact case-insensitive match on primary name
                if ch.get("name", "").lower() == name_lower:
                    return copy.deepcopy(ch)
                # Exact case-insensitive match on any alias
                aliases = ch.get("aliases", [])
                for alias in aliases:
                    if alias.lower() == name_lower:
                        return copy.deepcopy(ch)
            return None

    async def update(
        self, character_id: str, user_id: str, update_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            ch = self._characters.get(character_id)
            if not ch or ch.get("user_id") != user_id:
                return None

            for key, val in update_data.items():
                if val is not None and key not in (
                    "character_id", "story_id", "user_id", "created_at"
                ):
                    ch[key] = copy.deepcopy(val)

            ch["updated_at"] = datetime.now(timezone.utc)
            return copy.deepcopy(ch)

    async def delete_by_story(self, story_id: str, user_id: str) -> int:
        with self._lock:
            char_ids = [
                cid
                for cid, ch in self._characters.items()
                if ch.get("story_id") == story_id and ch.get("user_id") == user_id
            ]
            for cid in char_ids:
                del self._characters[cid]

            rel_ids = [
                rid
                for rid, rel in self._relationships.items()
                if rel.get("story_id") == story_id and rel.get("user_id") == user_id
            ]
            for rid in rel_ids:
                del self._relationships[rid]

            return len(char_ids)

    async def save_relationship(self, rel_data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            doc = copy.deepcopy(rel_data)
            self._relationships[doc["relationship_id"]] = doc
            return copy.deepcopy(doc)

    async def list_relationships(self, story_id: str, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                copy.deepcopy(rel)
                for rel in self._relationships.values()
                if rel.get("story_id") == story_id and rel.get("user_id") == user_id
            ]

    async def find_relationship(
        self, story_id: str, user_id: str, source_id: str, target_id: str, rel_type: str
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            for rel in self._relationships.values():
                if (
                    rel.get("story_id") == story_id
                    and rel.get("user_id") == user_id
                    and rel.get("source_character_id") == source_id
                    and rel.get("target_character_id") == target_id
                    and rel.get("relationship_type") == rel_type
                ):
                    return copy.deepcopy(rel)
            return None

    def clear(self) -> None:
        """Clear all storage (for testing)."""
        with self._lock:
            self._characters.clear()
            self._relationships.clear()
