"""Phase 6 — Model strategy for the agent runtime.

Selects *how* to use models for a given turn: which capability lane, which
cost/latency profile, which timeout, and which fallback posture — without
hard-coding vendor names into the loop.

The existing ModelRouter / KimiDeepSeekRouter remain the transport layer.
This module is pure policy + diagnostics.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

StrategyProfile = Literal["economy", "balanced", "quality", "failover_aggressive"]
LaneName = Literal[
    "fast",
    "planner",
    "coder",
    "researcher",
    "browser",
    "document",
    "spreadsheet",
    "verifier",
]

# Default wall-clock timeout per lane (seconds). Overridable via env
# MODEL_TIMEOUT_<LANE> (e.g. MODEL_TIMEOUT_PLANNER=90).
_DEFAULT_TIMEOUTS: dict[str, int] = {
    "fast": 20,
    "planner": 60,
    "coder": 90,
    "researcher": 90,
    "browser": 60,
    "document": 60,
    "spreadsheet": 60,
    "verifier": 45,
}

# Role strings understood by the product router.
_LANE_TO_ROUTER_ROLE: dict[str, str] = {
    "fast": "fast",
    "planner": "planning",
    "coder": "coder",
    "researcher": "researcher",
    "browser": "browser",
    "document": "planning",
    "spreadsheet": "planning",
    "verifier": "verification",
}

# Preferred fallback chain names (router keys) per lane.
_LANE_FALLBACK_CHAINS: dict[str, list[str]] = {
    "fast": ["deepseek", "kimi", "nvidia", "planner"],
    "planner": ["deepseek", "planner", "nvidia", "kimi"],
    "coder": ["planner", "nvidia", "kimi", "deepseek"],
    "researcher": ["planner", "nvidia", "kimi", "deepseek"],
    "browser": ["nvidia", "kimi", "planner", "deepseek"],
    "document": ["planner", "nvidia", "kimi", "deepseek"],
    "spreadsheet": ["planner", "nvidia", "kimi", "deepseek"],
    "verifier": ["kimi", "deepseek", "nvidia", "planner"],
}


def _timeout_for_lane(lane: str) -> int:
    env_key = f"MODEL_TIMEOUT_{lane.upper()}"
    raw = os.environ.get(env_key)
    if raw:
        try:
            return max(5, int(raw))
        except ValueError:
            pass
    return int(_DEFAULT_TIMEOUTS.get(lane, 60))


@dataclass(frozen=True)
class StrategyDecision:
    """Immutable decision for one model call."""

    lane: str
    profile: StrategyProfile
    router_role: str
    timeout_s: int
    fallback_chain: tuple[str, ...]
    temperature: float
    max_tokens: int
    prefer_tools: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fallback_chain"] = list(self.fallback_chain)
        d["reasons"] = list(self.reasons)
        return d


@dataclass
class TaskSignals:
    """Lightweight signals derived from the current run / turn."""

    goal: str = ""
    needs_vision: bool = False
    needs_browser: bool = False
    needs_code: bool = False
    needs_research: bool = False
    needs_spreadsheet: bool = False
    needs_document: bool = False
    is_verification: bool = False
    is_classification: bool = False
    step_count: int = 0
    research_depth: str | None = None  # quick | standard | deep
    explicit_lane: str | None = None
    explicit_profile: StrategyProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_signals(
    *,
    goal: str = "",
    tools_used: list[str] | None = None,
    research_depth: str | None = None,
    step_count: int = 0,
    role_hint: str | None = None,
) -> TaskSignals:
    """Heuristic signal extraction from goal text and recent tools."""
    g = (goal or "").lower()
    tools = {str(t).lower() for t in (tools_used or [])}

    needs_browser = any(t.startswith("browser") or t in ("computer", "navigate") for t in tools)
    needs_browser = needs_browser or any(
        k in g for k in ("click", "website", "web page", "browser", "screenshot", "login to")
    )
    needs_code = any(t in tools for t in ("python_exec", "shell_exec", "file_edit", "file_write"))
    needs_code = needs_code or any(
        k in g for k in ("code", "python", "debug", "refactor", "repository", "pull request")
    )
    needs_research = any(t in tools for t in ("web_search", "search", "fetch_url", "http_get"))
    needs_research = needs_research or any(
        k in g for k in ("research", "compare", "sources", "cite", "find out", "investigate")
    )
    needs_spreadsheet = any(k in g for k in ("spreadsheet", "excel", "csv", "xlsx", "formula"))
    needs_document = any(k in g for k in ("docx", "document", "pptx", "slide", "pdf report"))
    needs_vision = "vision" in tools or "screenshot" in g

    is_verification = (role_hint or "") in ("verifier", "verification", "summarizer")
    is_classification = (role_hint or "") in ("fast", "intent", "title") or (
        len(g) < 80 and any(k in g for k in ("classify", "label", "title", "summarize briefly"))
    )

    explicit_lane = None
    if role_hint in _LANE_TO_ROUTER_ROLE:
        explicit_lane = role_hint

    return TaskSignals(
        goal=goal or "",
        needs_vision=needs_vision,
        needs_browser=needs_browser,
        needs_code=needs_code,
        needs_research=needs_research,
        needs_spreadsheet=needs_spreadsheet,
        needs_document=needs_document,
        is_verification=is_verification,
        is_classification=is_classification,
        step_count=int(step_count or 0),
        research_depth=research_depth,
        explicit_lane=explicit_lane,
    )


def select_strategy(
    signals: TaskSignals | None = None,
    *,
    profile: StrategyProfile | None = None,
) -> StrategyDecision:
    """Choose lane + timeouts + fallback posture for this turn."""
    signals = signals or TaskSignals()
    profile = (
        signals.explicit_profile
        or profile
        or _profile_from_env()
        or "balanced"
    )
    reasons: list[str] = []

    # 1) Lane selection
    if signals.explicit_lane and signals.explicit_lane in _LANE_TO_ROUTER_ROLE:
        lane = signals.explicit_lane
        reasons.append(f"explicit_lane:{lane}")
    elif signals.is_verification:
        lane = "verifier"
        reasons.append("verification_role")
    elif signals.is_classification:
        lane = "fast"
        reasons.append("classification")
    elif signals.needs_vision or signals.needs_browser:
        lane = "browser"
        reasons.append("browser_or_vision")
    elif signals.needs_code:
        lane = "coder"
        reasons.append("code_signals")
    elif signals.needs_spreadsheet:
        lane = "spreadsheet"
        reasons.append("spreadsheet_signals")
    elif signals.needs_document:
        lane = "document"
        reasons.append("document_signals")
    elif signals.needs_research or (signals.research_depth in ("standard", "deep")):
        lane = "researcher"
        reasons.append("research_signals")
    else:
        lane = "planner"
        reasons.append("default_planner")

    # 2) Profile adjustments
    temperature = 0.2
    max_tokens = 4096
    prefer_tools = lane not in ("fast", "verifier")

    if profile == "economy":
        temperature = 0.1
        max_tokens = 2048
        # Prefer cheaper lanes when possible
        if lane == "planner" and signals.is_classification:
            lane = "fast"
            reasons.append("economy_downgrade_fast")
        reasons.append("profile:economy")
    elif profile == "quality":
        temperature = 0.3
        max_tokens = 8192
        reasons.append("profile:quality")
    elif profile == "failover_aggressive":
        reasons.append("profile:failover_aggressive")
    else:
        reasons.append("profile:balanced")

    # Deep research gets more tokens
    if signals.research_depth == "deep":
        max_tokens = max(max_tokens, 8192)
        reasons.append("depth:deep")

    timeout_s = _timeout_for_lane(lane)
    if profile == "failover_aggressive":
        # Shorter per-attempt timeout so failover kicks in sooner
        timeout_s = max(10, int(timeout_s * 0.6))

    chain = tuple(_LANE_FALLBACK_CHAINS.get(lane, ["planner", "deepseek", "kimi", "nvidia"]))
    if profile == "economy":
        # Prefer deepseek first when present
        if "deepseek" in chain:
            chain = ("deepseek",) + tuple(c for c in chain if c != "deepseek")
            reasons.append("economy_prefer_deepseek")

    return StrategyDecision(
        lane=lane,
        profile=profile,  # type: ignore[arg-type]
        router_role=_LANE_TO_ROUTER_ROLE.get(lane, "planning"),
        timeout_s=timeout_s,
        fallback_chain=chain,
        temperature=temperature,
        max_tokens=max_tokens,
        prefer_tools=prefer_tools,
        reasons=tuple(reasons),
    )


def _profile_from_env() -> StrategyProfile | None:
    raw = (os.environ.get("MODEL_STRATEGY_PROFILE") or "").strip().lower()
    if raw in ("economy", "balanced", "quality", "failover_aggressive"):
        return raw  # type: ignore[return-value]
    return None


@dataclass
class StrategyLedger:
    """In-memory log of strategy decisions for diagnostics / tests."""

    decisions: list[StrategyDecision] = field(default_factory=list)

    def record(self, decision: StrategyDecision) -> None:
        self.decisions.append(decision)

    def last(self) -> StrategyDecision | None:
        return self.decisions[-1] if self.decisions else None

    def summary(self) -> dict[str, Any]:
        by_lane: dict[str, int] = {}
        for d in self.decisions:
            by_lane[d.lane] = by_lane.get(d.lane, 0) + 1
        return {
            "count": len(self.decisions),
            "by_lane": by_lane,
            "last": self.last().to_dict() if self.last() else None,
        }


# Process-local ledger (optional observers).
_default_ledger = StrategyLedger()


def get_strategy_ledger() -> StrategyLedger:
    return _default_ledger


def decide_and_record(
    signals: TaskSignals | None = None,
    *,
    profile: StrategyProfile | None = None,
    ledger: StrategyLedger | None = None,
) -> StrategyDecision:
    decision = select_strategy(signals, profile=profile)
    (ledger or _default_ledger).record(decision)
    return decision


__all__ = [
    "StrategyProfile",
    "StrategyDecision",
    "TaskSignals",
    "StrategyLedger",
    "infer_signals",
    "select_strategy",
    "decide_and_record",
    "get_strategy_ledger",
]
