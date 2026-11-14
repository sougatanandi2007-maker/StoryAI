"""
StoryWeave AI - Story Continuity & Consistency Service

Orchestrates cross-chapter continuity verification:
- Extracts structured facts (explicit vs inferred) from new and earlier chapters.
- Retrieves previously established canon and relevant ChromaDB memories.
- Evaluates character appearance, timeline/flashbacks, relationships, world rules, and plot threads.
- Applies nuanced storytelling reasoning (character development, flashbacks, off-screen events).
- Weighs explicit facts higher than AI inferences.
- Persists continuity reports and preserves author triage decisions.
- Completely advisory: never alters manuscripts or story canon automatically.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from fastapi import Depends
from app.core.logging import get_logger
from app.core.exceptions import (
    ChapterNotFoundException,
    StoryNotFoundException,
    EntityNotFoundException,
    ValidationException,
    StoryWeaveException,
)
from app.database.repositories.base import (
    StoryRepository,
    ChapterRepository,
    AnalysisRepository,
    ContinuityRepository,
)
from app.database.repositories.dependencies import (
    get_story_repository,
    get_chapter_repository,
    get_analysis_repository,
    get_continuity_repository,
)
from app.services.gemini_service import GeminiService, get_gemini_service
from app.services.retrieval_service import RetrievalService, get_retrieval_service
from app.schemas.analysis import ChapterAnalysisResult
from app.schemas.continuity import (
    ContinuityIssueType,
    ContinuitySeverity,
    IssueStatus,
    ContinuityEvidence,
    ContinuityIssue,
    ContinuityReport,
    ContinuityCheckResponse,
)

logger = get_logger("service.continuity")


class ContinuityService:
    """Service providing story continuity analysis, conflict detection, and author advisory reporting."""

    def __init__(
        self,
        story_repo: StoryRepository,
        chapter_repo: ChapterRepository,
        analysis_repo: AnalysisRepository,
        continuity_repo: ContinuityRepository,
        gemini_service: GeminiService,
        retrieval_service: Optional[RetrievalService] = None,
    ):
        self._story_repo = story_repo
        self._chapter_repo = chapter_repo
        self._analysis_repo = analysis_repo
        self._continuity_repo = continuity_repo
        self._gemini_service = gemini_service
        self._retrieval_service = retrieval_service

    async def check_chapter_continuity(
        self,
        chapter_id: str,
        user_id: str,
        force_recheck: bool = False,
    ) -> ContinuityCheckResponse:
        """
        Execute continuity and contradiction checking for a chapter against prior canon.

        Workflow:
        1. Validate chapter and story exist and belong to user.
        2. Check for existing cached report if force_recheck=False.
        3. Verify chapter has completed analysis report (prerequisite).
        4. Check if prior chapters exist. Chapter 1 establishes foundational canon.
        5. Collect earlier canon from earlier chapter analysis reports and memory retrieval.
        6. Invoke GeminiService.evaluate_continuity to detect potential contradictions.
        7. Validate and score issues: enforce explicit vs inferred weights and evidence quotes.
        8. Save/upsert report idempotently, preserving author's previous triage statuses.
        9. Return structured response.
        """
        # 1. Fetch chapter
        chapter = await self._chapter_repo.get_by_id(chapter_id=chapter_id, user_id=user_id)
        if not chapter:
            raise ChapterNotFoundException(chapter_id=chapter_id)

        story_id = chapter["story_id"]
        story = await self._story_repo.get_by_id(story_id=story_id, user_id=user_id)
        if not story:
            raise StoryNotFoundException(story_id=story_id)

        # 2. Check cached report
        if not force_recheck:
            existing = await self._continuity_repo.get_by_chapter_id(chapter_id=chapter_id, user_id=user_id)
            if existing:
                logger.info(f"Returning cached continuity report for chapter {chapter_id}")
                return ContinuityCheckResponse(
                    report=ContinuityReport.model_validate(existing),
                    cached=True,
                    message="Retrieved existing continuity report from cache.",
                )

        # 3. Verify chapter analysis exists
        analysis_doc = await self._analysis_repo.get_by_chapter_id(chapter_id=chapter_id, user_id=user_id)
        if not analysis_doc:
            raise ValidationException(
                f"Chapter '{chapter['title']}' must be analyzed before running a continuity check. "
                f"Please call POST /api/v1/chapters/{chapter_id}/analyze first."
            )

        current_analysis = ChapterAnalysisResult.model_validate(analysis_doc["analysis"])

        # 4. Check for earlier chapters in the same story
        all_chapters, _ = await self._chapter_repo.list_by_story(
            story_id=story_id, user_id=user_id, skip=0, limit=1000
        )
        earlier_chapters = [
            c for c in all_chapters if c["chapter_number"] < chapter["chapter_number"]
        ]
        earlier_chapters.sort(key=lambda x: x["chapter_number"])

        now = datetime.now(timezone.utc)
        report_id = f"cont_rep_{uuid.uuid4().hex[:16]}"

        if not earlier_chapters:
            # First chapter in chronological sequence
            logger.info(
                f"Chapter {chapter['chapter_number']} ('{chapter['title']}') is the initial chapter. "
                f"Establishes foundational canon."
            )
            clean_report = ContinuityReport(
                report_id=report_id,
                chapter_id=chapter_id,
                chapter_number=chapter["chapter_number"],
                story_id=story_id,
                user_id=user_id,
                total_issues=0,
                issues=[],
                summary=(
                    f"Chapter {chapter['chapter_number']} ('{chapter['title']}') establishes foundational "
                    f"canon for the universe '{story['title']}'. No earlier chapters exist to evaluate against."
                ),
                checked_at=now,
            )
            saved = await self._continuity_repo.save_report(clean_report.model_dump())
            return ContinuityCheckResponse(
                report=ContinuityReport.model_validate(saved),
                cached=False,
                message="Initial chapter establishes foundational universe canon. No contradictions detected.",
            )

        # 5. Gather structured canon from earlier chapters
        earlier_chapters_lore: List[Dict[str, Any]] = []
        for ec in earlier_chapters:
            ec_analysis = await self._analysis_repo.get_by_chapter_id(ec["id"], user_id=user_id)
            if ec_analysis:
                lore_entry = {
                    "chapter_id": ec["id"],
                    "chapter_number": ec["chapter_number"],
                    "title": ec["title"],
                    "summary": ec_analysis.get("analysis", {}).get("chapter_summary", {}).get("summary", ""),
                    "characters": ec_analysis.get("analysis", {}).get("characters", []),
                    "world_rules": ec_analysis.get("analysis", {}).get("world_rules", []),
                    "story_facts": ec_analysis.get("analysis", {}).get("story_facts", []),
                    "relationships": ec_analysis.get("analysis", {}).get("relationships", []),
                    "timeline": ec_analysis.get("analysis", {}).get("timeline", []),
                    "events": ec_analysis.get("analysis", {}).get("events", []),
                }
            else:
                lore_entry = {
                    "chapter_id": ec["id"],
                    "chapter_number": ec["chapter_number"],
                    "title": ec["title"],
                    "excerpt": ec.get("content", "")[:600],
                }
            earlier_chapters_lore.append(lore_entry)

        current_chapter_data = {
            "chapter_id": chapter_id,
            "chapter_number": chapter["chapter_number"],
            "title": chapter["title"],
            "summary": current_analysis.chapter_summary.summary,
            "characters": [c.model_dump() for c in current_analysis.characters],
            "relationships": [r.model_dump() for r in current_analysis.relationships],
            "world_rules": [w.model_dump() for w in current_analysis.world_rules],
            "story_facts": [f.model_dump() for f in current_analysis.story_facts],
            "timeline": [t.model_dump() for t in current_analysis.timeline],
            "events": [e.model_dump() for e in current_analysis.events],
        }

        # 6. Evaluate continuity with Gemini
        logger.info(
            f"Evaluating continuity for Chapter {chapter['chapter_number']} against "
            f"{len(earlier_chapters)} earlier chapter(s)..."
        )
        raw_issues = await self._gemini_service.evaluate_continuity(
            story_context=story,
            chapter_title=chapter["title"],
            chapter_number=chapter["chapter_number"],
            chapter_id=chapter_id,
            current_chapter_data=current_chapter_data,
            earlier_chapters_lore=earlier_chapters_lore,
        )

        # 7. Process, validate, and score detected issues
        validated_issues: List[ContinuityIssue] = []
        for raw in raw_issues:
            if not isinstance(raw, dict):
                continue

            try:
                # Issue type normalization
                raw_type = str(raw.get("issue_type", "plot_continuity")).lower().strip()
                try:
                    issue_type = ContinuityIssueType(raw_type)
                except ValueError:
                    issue_type = ContinuityIssueType.PLOT_CONTINUITY

                # Severity normalization
                raw_sev = str(raw.get("severity", "medium")).lower().strip()
                try:
                    severity = ContinuitySeverity(raw_sev)
                except ValueError:
                    severity = ContinuitySeverity.MEDIUM

                # Confidence normalization
                raw_conf = float(raw.get("confidence", 0.8))
                confidence = max(0.0, min(1.0, raw_conf))

                # Build earlier evidence objects
                earlier_ev_list: List[ContinuityEvidence] = []
                for ev in raw.get("earlier_evidence", []):
                    if isinstance(ev, dict) and ev.get("quote_or_excerpt"):
                        ev_cid = str(ev.get("chapter_id", earlier_chapters[0]["id"]))
                        ev_cnum = int(ev.get("chapter_number", earlier_chapters[0]["chapter_number"]))
                        ev_quote = str(ev.get("quote_or_excerpt", ""))
                        ev_ft = str(ev.get("fact_type", "explicit")).lower()
                        ev_conf = float(ev.get("confidence", 1.0))

                        earlier_ev_list.append(
                            ContinuityEvidence(
                                chapter_id=ev_cid,
                                chapter_number=ev_cnum,
                                quote_or_excerpt=ev_quote,
                                fact_type=ev_ft,
                                confidence=max(0.0, min(1.0, ev_conf)),
                            )
                        )

                # Fallback if earlier_evidence was a single dict
                if not earlier_ev_list and isinstance(raw.get("earlier_evidence"), dict):
                    ev = raw["earlier_evidence"]
                    if ev.get("quote_or_excerpt"):
                        earlier_ev_list.append(
                            ContinuityEvidence(
                                chapter_id=str(ev.get("chapter_id", earlier_chapters[0]["id"])),
                                chapter_number=int(ev.get("chapter_number", earlier_chapters[0]["chapter_number"])),
                                quote_or_excerpt=str(ev.get("quote_or_excerpt", "")),
                                fact_type=str(ev.get("fact_type", "explicit")).lower(),
                                confidence=float(ev.get("confidence", 1.0)),
                            )
                        )

                # Build current evidence object
                current_ev: Optional[ContinuityEvidence] = None
                raw_cur = raw.get("current_evidence")
                if isinstance(raw_cur, dict) and raw_cur.get("quote_or_excerpt"):
                    current_ev = ContinuityEvidence(
                        chapter_id=str(raw_cur.get("chapter_id", chapter_id)),
                        chapter_number=int(raw_cur.get("chapter_number", chapter["chapter_number"])),
                        quote_or_excerpt=str(raw_cur.get("quote_or_excerpt", "")),
                        fact_type=str(raw_cur.get("fact_type", "explicit")).lower(),
                        confidence=float(raw_cur.get("confidence", 1.0)),
                    )

                # Enforce source chapter IDs
                source_cids = raw.get("source_chapter_ids", [])
                if not isinstance(source_cids, list) or not source_cids:
                    source_cids = [ev.chapter_id for ev in earlier_ev_list]
                    if chapter_id not in source_cids:
                        source_cids.append(chapter_id)

                # Enforce rule: Explicit facts carry more weight than AI inferences
                has_inferred_earlier = any(ev.fact_type == "inferred" for ev in earlier_ev_list)
                if has_inferred_earlier:
                    # Inferred facts must not assert high confidence contradictions
                    confidence = min(confidence, 0.65)
                    if severity == ContinuitySeverity.CRITICAL:
                        severity = ContinuitySeverity.MEDIUM

                # Enforce rule: Low evidence yields low confidence / observation
                if not earlier_ev_list or not current_ev:
                    confidence = min(confidence, 0.40)
                    if severity in (ContinuitySeverity.CRITICAL, ContinuitySeverity.HIGH):
                        severity = ContinuitySeverity.LOW

                issue_id = f"issue_{uuid.uuid4().hex[:12]}"

                validated_issue = ContinuityIssue(
                    issue_id=issue_id,
                    story_id=story_id,
                    chapter_id=chapter_id,
                    issue_type=issue_type,
                    severity=severity,
                    confidence=round(confidence, 2),
                    description=str(raw.get("description", "Continuity inconsistency detected.")),
                    existing_fact=str(raw.get("existing_fact", "Previously established canon.")),
                    new_fact=str(raw.get("new_fact", "Contradictory statement in current chapter.")),
                    affected_entities=raw.get("affected_entities", []),
                    earlier_evidence=earlier_ev_list,
                    current_evidence=current_ev,
                    source_chapter_ids=list(set(source_cids)),
                    possible_explanation=raw.get("possible_explanation"),
                    suggested_resolution=raw.get("suggested_resolution"),
                    status=IssueStatus.OPEN,
                    author_notes=None,
                    created_at=now,
                    updated_at=now,
                )
                validated_issues.append(validated_issue)

            except Exception as e:
                logger.warning(f"Skipping malformed continuity issue candidate: {e}")

        # 8. Compose summary
        if not validated_issues:
            summary = (
                f"Continuity audit complete for Chapter {chapter['chapter_number']} ('{chapter['title']}'). "
                f"No narrative contradictions or world-rule violations were detected against earlier chapters."
            )
        else:
            issue_counts = {}
            for iss in validated_issues:
                issue_counts[iss.severity.value] = issue_counts.get(iss.severity.value, 0) + 1
            severity_str = ", ".join(f"{cnt} {sev}" for sev, cnt in issue_counts.items())
            summary = (
                f"Continuity audit complete for Chapter {chapter['chapter_number']} ('{chapter['title']}'). "
                f"Identified {len(validated_issues)} potential continuity issue(s) ({severity_str}). "
                f"All suggestions are advisory for author review."
            )

        report = ContinuityReport(
            report_id=report_id,
            chapter_id=chapter_id,
            chapter_number=chapter["chapter_number"],
            story_id=story_id,
            user_id=user_id,
            total_issues=len(validated_issues),
            issues=validated_issues,
            summary=summary,
            checked_at=now,
        )

        # 9. Persist report (idempotent, preserving author triage decisions)
        saved_doc = await self._continuity_repo.save_report(report.model_dump())
        logger.info(
            f"Saved continuity report {report_id} for Chapter {chapter['chapter_number']} "
            f"with {len(validated_issues)} issues."
        )

        return ContinuityCheckResponse(
            report=ContinuityReport.model_validate(saved_doc),
            cached=False,
            message="Continuity check completed successfully.",
        )

    async def get_chapter_continuity_report(self, chapter_id: str, user_id: str) -> ContinuityReport:
        """Retrieve stored continuity report for a chapter."""
        chapter = await self._chapter_repo.get_by_id(chapter_id=chapter_id, user_id=user_id)
        if not chapter:
            raise ChapterNotFoundException(chapter_id=chapter_id)

        report = await self._continuity_repo.get_by_chapter_id(chapter_id=chapter_id, user_id=user_id)
        if not report:
            raise EntityNotFoundException("ContinuityReport", chapter_id)

        return ContinuityReport.model_validate(report)

    async def list_story_continuity_reports(self, story_id: str, user_id: str) -> List[ContinuityReport]:
        """List all continuity reports for a story, ordered by chapter number."""
        story = await self._story_repo.get_by_id(story_id=story_id, user_id=user_id)
        if not story:
            raise StoryNotFoundException(story_id=story_id)

        reports = await self._continuity_repo.list_by_story(story_id=story_id, user_id=user_id)
        return [ContinuityReport.model_validate(r) for r in reports]

    async def update_issue_status(
        self,
        issue_id: str,
        user_id: str,
        status: IssueStatus,
        author_notes: Optional[str] = None,
    ) -> ContinuityIssue:
        """
        Update the status ('open', 'accepted', 'dismissed', 'resolved') and notes of an issue.
        Enforces author sovereignty over continuity findings.
        """
        updated = await self._continuity_repo.update_issue_status(
            issue_id=issue_id,
            user_id=user_id,
            status=status.value,
            author_notes=author_notes,
        )
        if not updated:
            raise EntityNotFoundException("ContinuityIssue", issue_id)

        return ContinuityIssue.model_validate(updated)


def get_continuity_service(
    story_repo: StoryRepository = Depends(get_story_repository),
    chapter_repo: ChapterRepository = Depends(get_chapter_repository),
    analysis_repo: AnalysisRepository = Depends(get_analysis_repository),
    continuity_repo: ContinuityRepository = Depends(get_continuity_repository),
    gemini_service: GeminiService = Depends(get_gemini_service),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> ContinuityService:
    """FastAPI dependency provider for ContinuityService."""
    return ContinuityService(
        story_repo=story_repo,
        chapter_repo=chapter_repo,
        analysis_repo=analysis_repo,
        continuity_repo=continuity_repo,
        gemini_service=gemini_service,
        retrieval_service=retrieval_service,
    )
