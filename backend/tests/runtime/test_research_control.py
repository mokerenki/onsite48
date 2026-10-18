"""Unit tests for research depth, finish gates, and saturation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.runtime.research_control import (
    apply_depth_budget_caps,
    complexity_multiplier,
    count_successful_retrievals,
    is_research_goal,
    is_retrieval_tool,
    plan_research,
    research_finish_blocked,
    saturation_nudge_message,
    select_research_depth,
)
from app.runtime.verifier import apply_verdict_to_terminal


def test_is_research_goal_compare():
    assert is_research_goal("research the difference between tennsa and manus")
    assert is_research_goal("compare Tennsa vs Manus")
    assert not is_research_goal("what time is it")


def test_depth_selection():
    assert select_research_depth("quick research overview of Tennsa") == "quick"
    assert select_research_depth("deep dive into Tennsa market") == "deep"
    assert select_research_depth("research the difference between a and b") == "quick"
    assert select_research_depth("hello") == "none"


def test_finish_blocked_without_retrievals():
    msg = research_finish_blocked(
        "research the difference between tennsa and manus",
        [],
    )
    assert msg is not None
    assert "retrieval" in msg.lower()


def test_finish_allowed_with_retrieval():
    kr = [{"tool": "web_search", "result": {"ok": True, "results": [{"title": "x"}]}}]
    assert research_finish_blocked("research tennsa", kr) is None


def test_saturation_nudge_once():
    kr = [
        {"tool": "web_search", "result": {"ok": True, "results": [i]}}
        for i in range(6)
    ]
    msg = saturation_nudge_message("research tennsa", kr, already_nudged=False)
    assert msg is not None
    assert saturation_nudge_message("research tennsa", kr, already_nudged=True) is None


def test_budget_caps_standard():
    steps, wall, plan = apply_depth_budget_caps(
        max_steps=100,
        max_wall=2000.0,
        goal="research the difference between a and b",
    )
    assert plan.depth == "quick"
    assert steps <= 18
    assert wall <= 240.0


def test_complexity_multiplier_not_double_on_research_alone():
    assert complexity_multiplier("research tennsa") == 1.0


def test_count_retrievals():
    kr = [
        {"tool": "web_search", "result": {"ok": True}},
        {"tool": "file_write", "result": {"ok": True}},
        {"tool": "open_page", "result": {"error": "fail"}},
    ]
    assert count_successful_retrievals(kr) == 1


def test_verifier_empty_research_retries():
    terminal = apply_verdict_to_terminal(
        "completed",
        {"verdict": "fail", "unmet_requirements": ["no sources"], "missing_evidence": []},
        "Some prose only answer",
        finish_attempts=1,
        goal="research the difference between tennsa and manus",
        key_results=[],
    )
    assert terminal["status"] == "_retry_finish"


def test_verifier_research_with_evidence_soft_completes():
    kr = [{"tool": "web_search", "result": {"ok": True, "results": [{"url": "http://x"}]}}]
    terminal = apply_verdict_to_terminal(
        "completed",
        {"verdict": "fail", "unmet_requirements": ["single source"], "missing_evidence": []},
        "Findings...",
        finish_attempts=1,
        goal="research tennsa",
        key_results=kr,
    )
    assert terminal["status"] == "completed_unverified"


def test_is_retrieval_tool():
    assert is_retrieval_tool("web_search")
    assert is_retrieval_tool("open_page")
    assert not is_retrieval_tool("file_write")
    assert not is_retrieval_tool("finish")
