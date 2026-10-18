"""Research depth modes, evidence gates, and saturation controls.

Prevents two failure modes:
1. Instant "completed" on research goals with zero retrieval tools.
2. Runaway research after the evidence ledger already answers the goal.

Depth modes (product P1):
- quick: 1–2 searches, light synthesis, lower step budget
- standard: default for "research/compare/difference" asks
- deep: thorough multi-source collection (explicit user ask)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

ResearchDepth = Literal["quick", "standard", "deep", "none"]

# Goal looks like research / comparison that must use retrieval tools.
_RESEARCH_GOAL = re.compile(
    r"\b(?:"
    r"research|look\s+up|search\s+for|find\s+out|"
    r"compare|comparison|difference\s+between|differentiate|contrast|"
    r"versus|\bvs\.?\b|"
    r"who\s+is|who\s+are|what\s+is|market\s+analysis|"
    r"competitor|vendors?|options?\s+for|list\s+the\s+top|"
    r"overview\s+of|tell\s+me\s+about|background\s+on"
    r")\b",
    re.IGNORECASE,
)

_DEEP_SIGNALS = re.compile(
    r"\b(?:"
    r"deep\s+dive|thorough|comprehensive|exhaustive|in[- ]depth|"
    r"full\s+report|detailed\s+research|everything\s+about|"
    r"as\s+much\s+as\s+possible|leave\s+no\s+stone"
    r")\b",
    re.IGNORECASE,
)

_QUICK_SIGNALS = re.compile(
    r"\b(?:"
    r"quick\s+(?:research|look|summary|overview)|"
    r"brief(?:ly)?|high[- ]level|tl;?dr|in\s+a\s+nutshell|"
    r"short\s+summary|just\s+the\s+basics"
    r")\b",
    re.IGNORECASE,
)

# Tool name fragments that count as retrieval evidence.
_RETRIEVAL_FRAGMENTS = (
    "web_search",
    "search",
    "tavily",
    "brave",
    "open_page",
    "browse",
    "browser_observe",
    "http_get",
    "fetch_url",
    "read_page",
    "wide_research",
)


@dataclass(frozen=True)
class ResearchPlan:
    is_research: bool
    depth: ResearchDepth
    min_retrievals_before_finish: int
    saturation_retrievals: int
    max_steps_cap: int | None
    max_wall_s_cap: float | None
    label: str
    branches: list[ResearchBranch] = field(default_factory=list)  # Phase 1: parallel facets


@dataclass(frozen=True)
class EvidenceItem:
    url: str
    title: str
    content: str
    relevance: str = "general"


@dataclass(frozen=True)
class ResearchBranch:
    id: str
    facet: str
    queries: list[str]
    min_sources: int
    priority: int = 1


@dataclass
class BranchResult:
    branch_id: str
    facet: str
    evidence: list[EvidenceItem]
    summary: str
    sources: list[str]
    retrieval_count: int
    status: Literal["completed", "partial", "failed"]


@dataclass
class SynthesisResult:
    merged_evidence: list[EvidenceItem]
    executive_summary: str
    by_facet: dict[str, BranchResult]
    total_sources: int


# Facet extraction patterns for parallel branch planning.
# Matches "compare X, Y, and Z" or "X, Y, and Z of <subject>" style goals.
_FACET_SPLIT = re.compile(
    r"\b(?:compare|differentiate|contrast)\s+([^?]+?)\s*(?:\n|$)",
    re.IGNORECASE,
)
_COMMA_AND = re.compile(r"\band\b|\bor\b", re.IGNORECASE)
_SUBJECT_MARKERS = (
    "of ", "for ", "in ", "the ", "among ", "between ", "across ",
)


def _extract_facets(goal: str) -> list[str]:
    """Extract research facets from a goal like 'compare pricing, features, and market position of X'.

    Falls back to keyword extraction if no explicit list is found.
    """
    text = (goal or "").strip()
    if not text:
        return []

    lowered = text.lower()

    # Try "compare X, Y, and Z" pattern
    m = _FACET_SPLIT.search(text)
    if m:
        list_part = m.group(1).strip()
        # Remove trailing subject marker (e.g. "of 5 competitors in AI")
        for marker in _SUBJECT_MARKERS:
            if marker in lowered and lowered.index(marker) < len(list_part) + 5:
                idx = list_part.lower().rfind(marker)
                if idx > 0:
                    list_part = list_part[:idx].strip()
                break
        facets = [f.strip() for f in re.split(r"[,]", list_part) if f.strip()]
        if len(facets) >= 2:
            # Remove trailing "and <last>" artifact
            cleaned = []
            for f in facets:
                f = re.sub(r"\band\s+", "", f).strip()
                f = re.sub(r"\bor\s+", "", f).strip()
                if f:
                    cleaned.append(f)
            if len(cleaned) >= 2:
                return cleaned

    # Try keyword-based extraction: look for known research facets in the goal
    known_facets = [
        "pricing", "cost", "price", "features", "capabilities",
        "functionality", "market position", "market share", "positioning",
        "competitors", "vendors", "comparison", "pros and cons",
        "strengths", "weaknesses", "advantages", "disadvantages",
        "performance", "accuracy", "speed", "reliability",
        "usability", "integration", "api", "support", "documentation",
        "security", "compliance", "ecosystem", "adoption",
    ]
    found = []
    for kw in known_facets:
        if kw in lowered:
            found.append(kw)
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for f in found:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def _build_queries(facet: str, goal: str, depth: ResearchDepth) -> list[str]:
    """Generate search queries for a given facet."""
    subject = _strip_facet_from_goal(facet, goal)
    n = {"quick": 1, "standard": 2, "deep": 4}.get(depth, 2)
    base = f"{facet} {subject}".strip()
    queries = [f"{base} overview"]
    if n >= 2:
        queries.append(f"{base} recent 2024 2025")
    if n >= 3:
        queries.append(f"{base} comparison analysis")
    if n >= 4:
        queries.append(f"{facet} best practices {subject}")
    return queries[:n]


def _strip_facet_from_goal(facet: str, goal: str) -> str:
    """Extract the subject portion of the goal (everything except the facet list)."""
    text = (goal or "").strip()
    # Remove the facet word itself and common intro verbs
    subject = re.sub(r"\b(?:compare|research|find|look\s+up|search\s+for|list\s+the\s+top|overview\s+of)\b", "", text, flags=re.IGNORECASE)
    subject = subject.strip(" ,.")
    # If goal has "X, Y, and Z of <subject>", extract <subject>
    for marker in _SUBJECT_MARKERS:
        idx = subject.lower().find(marker)
        if idx > 3:  # not at the very start
            subject = subject[idx + len(marker):].strip()
            break
    return subject or text


def generate_research_branches(goal: str, depth: ResearchDepth = "standard", min_sources_per_branch: int = 2) -> list[ResearchBranch]:
    """Generate parallel research branches from a research goal.

    Splits the goal into facet-specific branches that can be explored concurrently.
    """
    facets = _extract_facets(goal)
    if not facets or len(facets) < 2:
        # Single broad branch — no parallelism benefit
        return []

    min_sources_map = {"quick": 1, "standard": 2, "deep": 3}
    ms = min_sources_per_branch if min_sources_per_branch else min_sources_map.get(depth, 2)

    branches: list[ResearchBranch] = []
    for i, facet in enumerate(facets):
        branch = ResearchBranch(
            id=f"branch_{i+1:02d}",
            facet=facet,
            queries=_build_queries(facet, goal, depth),
            min_sources=ms,
            priority=1,
        )
        branches.append(branch)
    return branches


def is_research_goal(goal: str) -> bool:
    text = (goal or "").strip()
    if not text:
        return False
    return bool(_RESEARCH_GOAL.search(text))


def select_research_depth(goal: str, *, explicit_depth: str | None = None) -> ResearchDepth:
    """Pick quick | standard | deep | none from goal text and optional override."""
    if explicit_depth in ("quick", "standard", "deep", "none"):
        return explicit_depth  # type: ignore[return-value]
    text = (goal or "").strip()
    if not text:
        return "none"
    # Explicit depth phrases imply research intent even without the word "research".
    if _DEEP_SIGNALS.search(text):
        return "deep"
    if _QUICK_SIGNALS.search(text) and is_research_goal(text):
        return "quick"
    if _QUICK_SIGNALS.search(text) and len(text.split()) <= 12:
        return "quick"
    if not is_research_goal(text):
        return "none"
    # Short factual / comparison queries use quick depth for a fast
    # single-turn answer instead of the heavy standard pipeline.
    if len(text.split()) <= 20:
        return "quick"
    return "standard"


def plan_research(goal: str, *, explicit_depth: str | None = None) -> ResearchPlan:
    depth = select_research_depth(goal, explicit_depth=explicit_depth)
    if depth == "none":
        return ResearchPlan(
            is_research=False,
            depth="none",
            min_retrievals_before_finish=0,
            saturation_retrievals=0,
            max_steps_cap=None,
            max_wall_s_cap=None,
            label="not_research",
        )
    if depth == "quick":
        return ResearchPlan(
            is_research=True,
            depth="quick",
            min_retrievals_before_finish=1,
            saturation_retrievals=3,
            max_steps_cap=18,
            max_wall_s_cap=240.0,
            label="quick_research",
        )
    if depth == "deep":
        return ResearchPlan(
            is_research=True,
            depth="deep",
            min_retrievals_before_finish=2,
            saturation_retrievals=12,
            max_steps_cap=80,
            max_wall_s_cap=1200.0,
            label="deep_research",
        )
     # standard
    branches = generate_research_branches(goal, depth=depth) if _enable_branching() else []
    return ResearchPlan(
        is_research=True,
        depth="standard",
        min_retrievals_before_finish=1,
        saturation_retrievals=6,
        max_steps_cap=40,
        max_wall_s_cap=600.0,
        label="standard_research",
        branches=branches,
    )


def _enable_branching() -> bool:
    """Check if parallel branch expansion is enabled via env flag."""
    import os as _os
    return (_os.environ.get("ENABLE_PARALLEL_RESEARCH", "").lower() in ("1", "true", "yes"))


def merge_branch_results(branches: list[BranchResult]) -> SynthesisResult:
    """Merge parallel branch results into a synthesis.

    Deduplicates evidence by URL, merges all sources, and orders branches
    by priority for the executive summary.
    """
    seen_urls: set[str] = set()
    merged: list[EvidenceItem] = []
    all_sources: list[str] = []
    by_facet: dict[str, BranchResult] = {}

    # Sort by priority descending (higher priority first)
    sorted_branches = sorted(branches, key=lambda b: getattr(b, "priority", 0), reverse=True)

    for br in branches:
        by_facet[br.facet] = br
        for ev in br.evidence:
            url_key = ev.url.lower().strip()
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)
            merged.append(ev)
        for src in br.sources:
            if src and src not in all_sources:
                all_sources.append(src)

    # Build executive summary from branch summaries
    parts = []
    for br in sorted_branches:
        if br.status in ("completed", "partial") and br.summary:
            parts.append(f"[{br.facet}] {br.summary}")
    executive_summary = "\n\n".join(parts) if parts else "No findings collected."

    return SynthesisResult(
        merged_evidence=merged,
        executive_summary=executive_summary,
        by_facet=by_facet,
        total_sources=len(all_sources),
    )


def is_retrieval_tool(name: str) -> bool:
    lowered = (name or "").lower()
    if not lowered:
        return False
    if lowered in ("finish", "ask_user", "file_write", "file_read", "file_edit"):
        return False
    return any(frag in lowered for frag in _RETRIEVAL_FRAGMENTS)


def count_successful_retrievals(key_results: Iterable[dict[str, Any]]) -> int:
    """Count successful retrieval tool results in the run ledger."""
    n = 0
    for entry in key_results or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("tool") or entry.get("name") or "")
        if not is_retrieval_tool(name):
            continue
        result = entry.get("result")
        if _looks_failed(result):
            continue
        n += 1
    return n


def _looks_failed(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, dict):
        if result.get("ok") is False:
            return True
        if result.get("error") and not result.get("results") and not result.get("content"):
            return True
        status = str(result.get("status") or "").lower()
        if status in ("error", "failed", "denied"):
            return True
    return False


def weighted_retrieval_increment(tool_name: str, result: Any) -> int:
    """Retrieval contribution of one tool result.

    wide_research internally runs N branch searches, so it counts as N
    retrievals toward research finish/saturation thresholds instead of 1.
    """
    if not is_retrieval_tool(tool_name) or _looks_failed(result):
        return 0
    if tool_name == "wide_research" and isinstance(result, dict):
        agent_count = result.get("agent_count")
        if isinstance(agent_count, int) and agent_count > 1:
            return min(agent_count, 12)
    return 1


def weighted_retrieval_count(key_results: Iterable[dict[str, Any]]) -> int:
    """Weighted retrieval total over a run ledger (replay-safe)."""
    return sum(
        weighted_retrieval_increment(str(e.get("tool") or e.get("name") or ""), e.get("result"))
        for e in key_results or []
        if isinstance(e, dict)
    )


def verifier_evidence(key_results: list[dict[str, Any]], max_entries: int = 60) -> list[dict[str, Any]]:
    """Retrieval-aware evidence selection for the completion verifier.

    A blind tail-window (key_results[-20:]) drops search results as the ledger
    grows during long research runs, making the verifier reject finishes that
    had ample evidence. Keep every retrieval entry; fill the rest with the
    most recent non-retrieval entries.
    """
    if len(key_results) <= max_entries:
        return key_results
    retrievals = [
        e for e in key_results
        if isinstance(e, dict) and is_retrieval_tool(str(e.get("tool") or e.get("name") or ""))
    ]
    if len(retrievals) >= max_entries:
        return retrievals[-max_entries:]
    retrieval_ids = {id(e) for e in retrievals}
    others = [e for e in key_results if id(e) not in retrieval_ids]
    return retrievals + others[-(max_entries - len(retrievals)):]


def finish_gate_exhausted(finish_attempts: int, retrieval_count: int, min_retrievals: int) -> bool:
    """True when the model keeps trying to finish a research goal without retrieval."""
    return finish_attempts >= 3 and retrieval_count < max(1, int(min_retrievals or 1))


def research_finish_blocked(
    goal: str,
    key_results: list[dict[str, Any]],
    *,
    explicit_depth: str | None = None,
    retrieval_count: int | None = None,
) -> str | None:
    """Return a rejection message if research finish is premature, else None."""
    plan = plan_research(goal, explicit_depth=explicit_depth)
    if not plan.is_research:
        return None
    n = count_successful_retrievals(key_results) if retrieval_count is None else max(0, int(retrieval_count))
    if n < plan.min_retrievals_before_finish:
        return (
            f"Research goals require at least {plan.min_retrievals_before_finish} "
            f"successful retrieval tool call(s) (web_search/open_page/browser) before finish. "
            f"Found {n}. Search or open sources, then finish with evidence_refs."
        )
    return None


def saturation_nudge_message(
    goal: str,
    key_results: list[dict[str, Any]],
    *,
    already_nudged: bool,
    explicit_depth: str | None = None,
    retrieval_count: int | None = None,
) -> str | None:
    """If retrieval count hit saturation and we have not nudged yet, return nudge text."""
    if already_nudged:
        return None
    plan = plan_research(goal, explicit_depth=explicit_depth)
    if not plan.is_research or plan.saturation_retrievals <= 0:
        return None
    n = count_successful_retrievals(key_results) if retrieval_count is None else max(0, int(retrieval_count))
    if n < plan.saturation_retrievals:
        return None
    return (
        f"[Research control] You already have {n} successful retrieval results "
        f"(saturation threshold for depth={plan.depth} is {plan.saturation_retrievals}). "
        "Synthesize findings with sources into the user-facing answer and call finish now, "
        "or name ONE specific missing gap and do at most one more targeted retrieval. "
        "Do not start a new broad search wave."
    )


def apply_depth_budget_caps(
    *,
    max_steps: int,
    max_wall: float,
    goal: str,
    explicit_depth: str | None = None,
    explicit_steps: bool = False,
    explicit_wall: bool = False,
) -> tuple[int, float, ResearchPlan]:
    """Cap step/wall budgets for research depth; never raises explicit caller budgets."""
    plan = plan_research(goal, explicit_depth=explicit_depth)
    steps = max_steps
    wall = max_wall
    if plan.is_research and plan.max_steps_cap is not None and not explicit_steps:
        steps = min(steps, plan.max_steps_cap)
    if plan.is_research and plan.max_wall_s_cap is not None and not explicit_wall:
        wall = min(wall, plan.max_wall_s_cap)
    return steps, wall, plan


def complexity_multiplier(goal: str) -> float:
    """Milder adaptive scale than the old always-2x research path."""
    goal_l = (goal or "").lower()
    hits = sum(
        1
        for k in (
            " and ",
            "report",
            "website",
            "deck",
            "spreadsheet",
            "analyze",
            "compare",
            "research",
            "build",
            "each of",
            "all of",
            "multi",
        )
        if k in goal_l
    )
    # Deep multi-deliverable goals still scale; pure "research X" does not 2x alone.
    if hits >= 4:
        return 1.75
    if hits >= 3:
        return 1.35
    return 1.0


__all__ = [
    "ResearchDepth",
    "ResearchPlan",
    "ResearchBranch",
    "BranchResult",
    "EvidenceItem",
    "SynthesisResult",
    "generate_research_branches",
    "merge_branch_results",
    "apply_depth_budget_caps",
    "complexity_multiplier",
    "count_successful_retrievals",
    "finish_gate_exhausted",
    "is_research_goal",
    "is_retrieval_tool",
    "plan_research",
    "research_finish_blocked",
    "saturation_nudge_message",
    "select_research_depth",
    "verifier_evidence",
    "weighted_retrieval_count",
    "weighted_retrieval_increment",
]
