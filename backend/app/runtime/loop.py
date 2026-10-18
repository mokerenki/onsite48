from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import structlog

from app.runtime.context import ContextBuilder
from app.runtime.journal import Journal, journal as default_journal
from app.runtime.models import complete as model_complete
from app.runtime.policy import (
    Decision,
    PolicyDecisionResult,
    evaluate as policy_evaluate,
    org_policy_from_schedule,
)
from app.runtime.recovery import (
    FailureClass,
    classify_failure,
    maybe_refresh_auth,
    recovery_message,
)
from app.runtime.tools import ToolSpec, build_tool_specs, specs_for_model
from app.runtime.verifier import (
    TERMINAL_STATES,
    apply_verdict_to_terminal,
    user_facing_message,
    verify_completion,
)
from app.runtime.list_answer import prefer_list_answer
from app.runtime.research_control import (
    BranchResult,
    EvidenceItem,
    ResearchBranch,
    apply_depth_budget_caps,
    complexity_multiplier,
    finish_gate_exhausted,
    is_research_goal,
    is_retrieval_tool,
    merge_branch_results,
    plan_research,
    research_finish_blocked,
    saturation_nudge_message,
    verifier_evidence,
    weighted_retrieval_count,
    weighted_retrieval_increment,
)

logger = structlog.get_logger("onsite48.runtime.loop")

import os as _os

def _env_int(name: str, default: int) -> int:
    try:
        return int(_os.environ.get(name, "") or default)
    except ValueError:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(_os.environ.get(name, "") or default)
    except ValueError:
        return default

# Global safety rails (per-run hard caps). Per-role working budgets live in
# roles_v2.py and typically sit below these. Raise via env, not code edits.
DEFAULT_MAX_STEPS = _env_int("AGENT_MAX_STEPS", 60)
DEFAULT_MAX_TOKENS = _env_int("AGENT_MAX_TOKENS", 400_000)
DEFAULT_MAX_COST_USD = _env_float("AGENT_MAX_COST_USD", 10.0)
DEFAULT_MAX_WALL_S = _env_float("AGENT_MAX_WALL_S", 1200)

DEFAULT_TOOL_TIMEOUT_S = _env_float("DEFAULT_TOOL_TIMEOUT_S", 120.0)

def _tool_timeout(spec: Any) -> float:
    """A tool without an explicit timeout must never wait forever."""
    raw = getattr(spec, "timeout_s", None)
    try:
        value = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        value = 0.0
    return value if value > 0 else DEFAULT_TOOL_TIMEOUT_S

# Progress heuristics. Static counters are a blunt instrument: verification
# loops (build→fix→rebuild) legitimately repeat calls. Budgets above are the
# real guard; these exist to catch true spinning early.
LOOP_SAME_CALL_LIMIT = _env_int("AGENT_SAME_CALL_LIMIT", 4)
LOOP_NO_PROGRESS_LIMIT = _env_int("AGENT_NO_PROGRESS_LIMIT", 8)

_LOOP_SAME_CALL_LIMIT_OVERRIDES: dict[str, int] = {
    # Execution verbs with genuine retry semantics get a wider berth.
    "python_exec": 8,
    "shell_exec": 8,
    "browser_act": 6,       # single page interactions can legitimately repeat
    "navigate": 5,
}

# ---------------------------------------------------------------------------
# Phase 1 feature flags — both default to false; enable via env after validation.
# ---------------------------------------------------------------------------
ENABLE_PARALLEL_RESEARCH = _os.environ.get("ENABLE_PARALLEL_RESEARCH", "").lower() in ("1", "true", "yes")
ENABLE_CONTEXT_SUMMARIZATION = _os.environ.get("ENABLE_CONTEXT_SUMMARIZATION", "").lower() in ("1", "true", "yes")
ENABLE_STRUCTURED_TOOL_RESULTS = _os.environ.get("ENABLE_STRUCTURED_TOOL_RESULTS", "").lower() in ("1", "true", "yes")
ENABLE_HIERARCHICAL_PLANS = _os.environ.get("ENABLE_HIERARCHICAL_PLANS", "").lower() in ("1", "true", "yes")

# Parallel research tunables
PARALLEL_BRANCH_CONCURRENCY = _env_int("PARALLEL_BRANCH_CONCURRENCY", 3)
MAX_PARALLEL_BRANCHES = _env_int("MAX_PARALLEL_BRANCHES", 5)

# Context summarization tunables
DEFAULT_SUMMARY_THRESHOLD = _env_int("CONTEXT_SUMMARY_THRESHOLD_TOKENS", 60_000)
DEFAULT_MAX_TOKENS_PHASE1 = _env_int("CONTEXT_MAX_TOKENS", 80_000)


def _default_finish_summary(state: Any, goal: str, key_results: list[dict[str, Any]]) -> str:
    """Derive a user-facing summary when the agent finishes without one.

    An agent that calls finish with an empty summary (for example, a website
    that was built and packaged without a written summary) must still deliver
    *some* answer to the user. This builds a short, honest summary from the goal
    and the most recent tool results instead of leaving the completion event
    with no message at all.
    """
    goal_text = (goal or "").strip()
    if not goal_text:
        return "The task has been completed."

    # Surface the most recent successful tool result when it carries a
    # human-readable message or artifact reference.
    recent = ""
    for entry in reversed(key_results or []):
        if not isinstance(entry, dict):
            continue
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        if result.get("error") or result.get("error_code"):
            continue
        message = str(result.get("message") or result.get("output") or "").strip()
        if message:
            recent = message
            break
        artifact = result.get("artifact")
        if isinstance(artifact, dict):
            name = str(artifact.get("name") or artifact.get("filename") or "").strip()
            if name:
                recent = f"Prepared artifact: {name}"
                break

    if recent:
        return f"{recent}"
    return "The task has been completed."


def _args_hash(args: dict) -> str:
    return hashlib.sha256(
        json.dumps(args, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _result_fingerprint(result: Any) -> str:
    """Stable, bounded fingerprint of a tool result for progress tracking."""
    if isinstance(result, dict):
        try:
            raw = json.dumps(result, sort_keys=True, default=str)
        except (TypeError, ValueError):
            raw = repr(result)
    else:
        raw = str(result)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:16]


def _structured_error(
    code: str, message: str, *, retryable: bool = False, suggestion: str = ""
) -> dict:
    return {
        "error_code": code,
        "message": message,
        "retryable": retryable,
        "suggestion": suggestion,
    }


def _exception_result(exc: Exception) -> dict[str, Any]:
    """Keep structured connector failures intact at the tool boundary.

    A provider failure used to be flattened to ``tool_error`` and then
    reclassified from whatever words appeared in its message. That permitted an
    unrelated browser action to enter connector OAuth recovery. Preserve the
    provider's safe machine fields instead; ordinary exceptions retain the
    generic tool-error contract.
    """
    try:
        from app.services.provider_errors import ProviderError

        if isinstance(exc, ProviderError):
            payload = exc.as_failure()
            payload["error_code"] = str(payload.pop("code", exc.code))
            return payload
    except Exception:
        # Failure serialization must never mask the original tool failure.
        pass
    return _structured_error("tool_error", str(exc), retryable=False)


async def _distill_fire_and_forget(*, run_id: str, goal: str, journal: Any,
                                   finish_payload: dict, org_id: str) -> None:
    """Phase 8: durable memory distillation. Never blocks the terminal event."""
    async def _run() -> None:
        try:
            from app.runtime.memory_distiller import distill_run, get_memory_store
            events = []
            try:
                events = await journal.list_events(run_id)
            except Exception:
                pass
            result = distill_run(
                run_id=run_id, goal=goal or "",
                finish_summary=str(finish_payload.get("summary") or "")[:2000],
                journal_events=events[-300:],
                verifier=finish_payload.get("verifier") if isinstance(finish_payload.get("verifier"), dict) else None,
            )
            store = get_memory_store()
            if hasattr(store, "upsert_many"):
                await store.upsert_many(result.records, org_id=org_id or "default")
        except Exception as exc:
            logger.debug("loop.distill_failed", run_id=run_id, error=str(exc)[:150])
    try:
        asyncio.create_task(_run())
    except Exception:
        pass


def _lessons_context_block(lessons: list) -> str | None:
    if not lessons:
        return None
    lines = ["[Prior learnings from completed tasks]"]
    for r in lessons[:5]:
        lines.append(f"- ({r.kind}) {r.text[:220]}")
    return "\n".join(lines)


async def _emit_live_event(event_emitter: Any, event: dict[str, Any]) -> None:
    """Forward a live UI event; a failed emission must never break the run."""
    if event_emitter is None:
        return
    try:
        emitted = event_emitter(event)
        if hasattr(emitted, "__await__"):
            await emitted
    except Exception as exc:
        logger.warning(
            "runtime.live_event_emit_failed",
            event_type=str(event.get("type")),
            error=str(exc)[:200],
        )


def _plan_step_description(name: str, args: dict[str, Any]) -> str:
    """A readable one-line label for a tool step in the live plan."""
    summary = ""
    for key in (
        "command", "cmd", "path", "file", "query", "url",
        "goal", "question", "message", "name",
    ):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            summary = value.strip()
            break
    if not summary:
        try:
            summary = json.dumps(args, default=str)
        except (TypeError, ValueError):
            summary = str(args)
    summary = summary.replace("\n", " ").strip()[:120]
    return f"{name}: {summary}" if summary else name


def _tool_result_output(result: Any) -> str:
    """Render a tool result as bounded text for the live step output."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result[:4000]
    try:
        return json.dumps(result, default=str)[:4000]
    except (TypeError, ValueError):
        return str(result)[:4000]


def _truncated_tool_args(args: Any) -> bool:
    """True when a tool call carries a large JSON payload the provider
    cut off at its response output limit.

    A truncated call arrives as ``{"_raw": "..."}`` once the model's
    tool-argument JSON fails to parse: it opens an object, streams a
    large string value, and never closes it. The raw string is long,
    starts with ``{`` and does not end with ``}`` — distinct from a
    merely malformed (short or properly closed) payload.
    """
    raw = args.get("_raw") if isinstance(args, dict) else None
    if not isinstance(raw, str):
        return False
    stripped = raw.strip()
    return len(stripped) > 500 and stripped.startswith("{") and not stripped.endswith("}")


def _auth_needs_input_payload(
    *,
    tool_name: str,
    result: Any,
    refresh: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a user-safe terminal response when connector access needs repair."""
    if isinstance(result, dict):
        message = str(result.get("user_message") or result.get("message") or "").strip()
        connector_id = str(result.get("connector_id") or "").strip()
        reconnect_required = bool(result.get("reconnect_required"))
    else:
        message = ""
        connector_id = ""
        reconnect_required = False
    subject = connector_id or tool_name
    if not message:
        message = f"The {subject} connection needs to be reconnected in Settings."
    if refresh and not refresh.get("refreshed") and not reconnect_required:
        message = f"{message} Automatic refresh was unavailable; reconnect the account in Settings."
    return {
        "summary": user_facing_message("needs_input", message),
        "status": "needs_input",
        "evidence_refs": [],
    }


# ---------------------------------------------------------------------------
# Phase 1: Context Window Management (sliding window + summarization)
# ---------------------------------------------------------------------------


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate: 1 token ≈ 4 characters (rough but consistent)."""
    total_chars = 0
    for msg in messages:
        for key in ("content", "summary", "question"):
            v = msg.get(key)
            if isinstance(v, str):
                total_chars += len(v)
        for key in ("tool_calls", "options"):
            v = msg.get(key)
            if isinstance(v, str):
                total_chars += len(v)
            elif isinstance(v, list):
                total_chars += len(json.dumps(v, default=str))
    return total_chars // 4


def extract_summary(result: Any) -> str:
    """Extract a 1-2 sentence summary from a tool result."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result[:300]
    if isinstance(result, dict):
        for key in ("message", "summary", "output", "text"):
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()[:300]
        for key in ("results", "items", "data", "hits"):
            v = result.get(key)
            if isinstance(v, list):
                return f"Returned {len(v)} results."
        if result.get("count") is not None:
            return f"Count: {result['count']}."
        if result.get("error") or result.get("error_code"):
            return f"Error: {str(result.get('error', result.get('error_code', '')))[:200]}"
        if result.get("ok") is True:
            return "Operation completed successfully."
    return str(result)[:300]


def extract_urls(result: Any) -> list[str]:
    """Extract URLs from a tool result for evidence tracking."""
    urls: list[str] = []
    if isinstance(result, dict):
        for key in ("url", "href", "link", "source_url", "canonical_url"):
            v = result.get(key)
            if isinstance(v, str) and v.startswith("http"):
                urls.append(v)
        for key in ("results", "sources", "items", "hits", "data"):
            v = result.get(key)
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        for uk in ("url", "href", "link"):
                            uv = item.get(uk)
                            if isinstance(uv, str) and uv.startswith("http") and uv not in urls:
                                urls.append(uv)
    elif isinstance(result, str) and result.startswith("http"):
        urls.append(result)
    return urls


class ContextManager:
    """Sliding-window context management with LLM summarization.

    After each model turn, if the estimated token count exceeds the summary
    threshold, older messages are collapsed into a compact summary using a
    fast (cheap) model. The most recent N turns are preserved verbatim.
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS_PHASE1,
        summary_threshold: int = DEFAULT_SUMMARY_THRESHOLD,
        summarizer_model: str = "fast",
        recent_turns: int = 6,
    ):
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold
        self.summarizer_model = summarizer_model
        self.recent_turns = recent_turns
        self._summarizer_prompt = (
            "Summarize the following conversation turns concisely. Preserve all"
            " key facts, decisions, tool results, URLs, and entity names."
            " Output only the summary text, no markdown headers."
        )

    async def maybe_summarize(self, state: Any) -> Any:
        """Check token count and summarize if over threshold. Mutates and returns state."""
        if not ENABLE_CONTEXT_SUMMARIZATION:
            return state

        messages = state.messages or []
        token_count = getattr(state, "token_count", 0) or 0
        estimated = estimate_tokens(messages)
        effective_tokens = max(token_count, estimated)

        # Also consider raw message count as a proxy
        if effective_tokens < self.summary_threshold and len(messages) < self.recent_turns * 3:
            return state

        # Partition: system messages stay, recent turn-pair window stays, older gets summarized
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= self.recent_turns:
            # Not enough older content to summarize — fall back to token-based check
            if effective_tokens < self.summary_threshold:
                return state

        recent = non_system[-self.recent_turns:] if len(non_system) > self.recent_turns else non_system[:]
        older = non_system[: max(0, len(non_system) - self.recent_turns)] if len(non_system) > self.recent_turns else []

        if not older:
            # Even if over threshold, nothing to summarize
            return state

        # Summarize older turns
        summary = await self._summarize_early_turns(older)

        new_messages = [
            *system_msgs,
            {"role": "system", "content": f"[Context summary: {summary}]"},
            *recent,
        ]

        state.messages = new_messages
        state.token_count = estimate_tokens(new_messages)
        state.summarized_turns = getattr(state, "summarized_turns", 0) + len(older)
        return state

    async def _summarize_early_turns(self, messages: list[dict[str, Any]]) -> str:
        """Use a fast/cheap model to summarize older conversation turns."""
        if not messages:
            return ""

        # Build a compact conversation for the summarizer
        content_parts = []
        for msg in messages:
            role = msg.get("role", "")
            text = msg.get("content", "")
            if isinstance(text, str) and text.strip():
                content_parts.append(f"[{role}] {text.strip()[:500]}")
            elif text:
                content_parts.append(f"[{role}] {str(text)[:500]}")

        prompt_text = "\n".join(content_parts[:50])  # cap input to summarizer

        summ_messages = [
            {"role": "system", "content": self._summarizer_prompt},
            {"role": "user", "content": prompt_text},
        ]

        try:
            result = await model_complete(
                summ_messages,
                tools=[],
                role="summarizer",
                tool_choice="none",
                temperature=0.1,
                max_tokens=512,
                model=self.summarizer_model,
            )
            return (result.content or "").strip()[:1000]
        except Exception:
            # Graceful fallback: naive truncation summary
            combined = " ".join(content_parts[:20])
            return combined[:500] if combined else "No prior context to summarize."


def _terminal_return(
    status: str,
    result: Any,
    state: Any,
    *,
    run_id: str | None = None,
    specs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a terminal loop.run payload with top-level metrics for evals/runner."""
    out: dict[str, Any] = {
        "status": status,
        "result": result,
        "state": state.to_dict() if hasattr(state, "to_dict") else (state or {}),
        "step_count": getattr(state, "step_count", 0) or 0,
        "token_count": getattr(state, "token_count", 0) or 0,
        "cost_usd": getattr(state, "cost_usd", None),
    }
    if run_id is not None:
        out["run_id"] = run_id
    # Include tool usage analytics
    if specs is not None:
        tool_stats = {}
        for name, spec in specs.items():
            if hasattr(spec, 'get_usage_stats'):
                stats = spec.get_usage_stats()
                if stats["call_count"] > 0:
                    tool_stats[name] = stats
        if tool_stats:
            out["tool_usage"] = tool_stats
    return out


def _is_error_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    return bool(result.get("error_code") or result.get("error"))


def _extract_untrusted_values(result: Any) -> list[str]:
    """Collect string tokens from untrusted results for provenance tracking."""
    vals: list[str] = []
    if not isinstance(result, dict):
        return vals
    for key in (
        "to", "email", "recipient", "url", "href", "address", "from",
        "content", "body", "text", "subject", "link",
    ):
        v = result.get(key)
        if isinstance(v, str) and len(v) >= 3:
            vals.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str) and len(item) >= 3:
                    vals.append(item)
    return vals


async def _run_readback(
    spec: ToolSpec,
    args: dict[str, Any],
    result: Any,
    *,
    run_id: str,
    journal: Journal,
) -> dict[str, Any] | None:
    """Execute ToolSpec.readback after a successful mutation. Journals the outcome."""
    if spec.readback is None:
        return None
    if spec.side_effect not in ("write", "destructive", "external_send", "spend"):
        return None
    if _is_error_result(result):
        return None
    try:
        rb_args = dict(args)
        if isinstance(result, dict):
            rb_args["_prior_result"] = result
        rb = await asyncio.wait_for(spec.readback(**rb_args), timeout=min(_tool_timeout(spec), 30))
        if not isinstance(rb, dict):
            rb = {"verified": False, "observed": {"raw": str(rb)[:500]}, "expected_diff": "non-dict readback"}
        await journal.append(
            run_id,
            "readback_result",
            {
                "tool": spec.name,
                "verified": rb.get("verified"),
                "observed": rb.get("observed"),
                "expected_diff": rb.get("expected_diff"),
            },
        )
        return rb
    except Exception as exc:
        rb = {
            "verified": False,
            "observed": {},
            "expected_diff": f"readback exception: {exc}",
        }
        await journal.append(
            run_id,
            "readback_result",
            {"tool": spec.name, **rb},
        )
        return rb


# ---------------------------------------------------------------------------
# Phase 1: Parallel research branch execution
# ---------------------------------------------------------------------------


async def _run_research_branch(
    branch: ResearchBranch,
    goal: str,
    specs: dict[str, ToolSpec],
    journal: Journal,
    run_id: str,
    *,
    model_override: str | None = None,
    event_emitter: Any = None,
    cancel_event: asyncio.Event | None = None,
    max_steps: int = 30,
    max_tokens: int = 60_000,
    max_cost_usd: float = 2.0,
    max_wall_s: float = 240.0,
) -> BranchResult:
    """Run a single research facet branch in isolation.

    Creates a sub-state with a facet-specific goal, runs a mini research loop
    (search → open → extract), and returns a BranchResult.
    """
    branch_run_id = f"{run_id}_branch_{branch.id}"
    branch_goal = (
        f"Research facet: {branch.facet}\n"
        f"Main goal: {goal}\n"
        f"Search queries to use: {json.dumps(branch.queries)}\n"
        f"Minimum sources required: {branch.min_sources}\n"
        "Gather evidence (URLs, titles, content snippets) for this facet."
        " Use web_search and open_page tools. When you have at least"
        f" {branch.min_sources} sources, call finish with a 2-3 sentence summary"
        " of findings and your evidence_refs."
    )

    # Use SpawnManager to run the branch as an isolated worker
    from app.runtime.spawn import SpawnManager
    from app.runtime.roles_v2 import RESEARCHER

    cancel_event = cancel_event or asyncio.Event()

    spawn_mgr = SpawnManager(
        parent_run_id=run_id,
        parent_role=RESEARCHER,
        workspace_root=Path(os.environ.get("AGENT_WORKSPACE", f"/tmp/onsite48_workspace/{run_id}")),
        loop_runner=None,  # We'll run inline instead
        journal=journal,
        depth=1,
        cancel_event=cancel_event,
    )

    # Run the branch using the parent loop runner (recursion into run())
    branch_result = await run(
        run_id=branch_run_id,
        goal=branch_goal,
        role_config={
            "name": "researcher",
            "system_prompt_file": "researcher_v2.txt",
            "tool_allowlist": RESEARCHER.tool_allowlist,
            "model_lane": "default",
            "budget": {"steps": max_steps, "tokens": max_tokens, "cost_usd": max_cost_usd, "wall_s": max_wall_s},
            "may_spawn": False,
            "depth_cap": 1,
            "max_summary_tokens": 800,
        },
        workspace=str(Path(os.environ.get("AGENT_WORKSPACE", f"/tmp/onsite48_workspace/{run_id}")) / "branches" / branch.id),
        journal=journal,
        model_override=model_override,
        max_steps=max_steps,
        max_tokens=max_tokens,
        max_cost_usd=max_cost_usd,
        max_wall_s=max_wall_s,
        parent_run_id=run_id,
        cancel_event=cancel_event,
    )

    # Extract evidence from the branch result
    evidence_items: list[EvidenceItem] = []
    sources: list[str] = []
    retrieval_count = 0

    # Pull evidence from the journal events for this branch
    try:
        branch_events = await journal.list_events(branch_run_id)
    except Exception:
        branch_events = []

    for ev in branch_events:
        et = ev.get("type") or ev.get("event_type") or ""
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        if et == "tool_call_result":
            r = payload.get("result", {})
            if isinstance(r, dict):
                if is_retrieval_tool(str(payload.get("name") or "")):
                    retrieval_count += 1
                    urls = extract_urls(r)
                    for url in urls:
                        if url not in sources:
                            sources.append(url)
                    title = str(r.get("title") or r.get("query") or "")[:200]
                    content = extract_summary(r)
                    if urls:
                        for url in urls:
                            evidence_items.append(EvidenceItem(
                                url=url,
                                title=title or f"Search result for {branch.facet}",
                                content=content or "",
                                relevance=branch.facet,
                            ))
                    elif content:
                        evidence_items.append(EvidenceItem(
                            url="",
                            title=title or f"{branch.facet} evidence",
                            content=content,
                            relevance=branch.facet,
                        ))

    # Extract summary from finish payload
    finish_payload = {}
    for ev in branch_events:
        et = ev.get("type") or ev.get("event_type") or ""
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        if et == "run_finished":
            finish_payload = payload
            break

    summary = finish_payload.get("summary", "") or branch_result.get("result", {}).get("summary", "")

    status = branch_result.get("status", "failed")
    if status not in ("completed", "completed_unverified", "partial"):
        status = "failed"

    return BranchResult(
        branch_id=branch.id,
        facet=branch.facet,
        evidence=evidence_items,
        summary=str(summary)[:1000],
        sources=sources,
        retrieval_count=retrieval_count,
        status=status,  # type: ignore
    )


async def run_research_branches_parallel(
    state: Any,
    branches: list[ResearchBranch],
    specs: dict[str, ToolSpec],
    journal: Journal,
    run_id: str,
    *,
    model_override: str | None = None,
    event_emitter: Any = None,
    cancel_event: asyncio.Event | None = None,
    branch_runner: Any = None,
) -> list[BranchResult]:
    """Run research branches concurrently with a semaphore limit.

    Each branch gets an isolated sub-run with a facet-specific goal.
    Returns BranchResults in the same order as branches.
    """
    if not branches:
        return []

    # Sort by priority (higher first) but keep original order for result mapping
    sorted_branches = sorted(branches, key=lambda b: getattr(b, "priority", 0), reverse=True)
    sem = asyncio.Semaphore(PARALLEL_BRANCH_CONCURRENCY)

    async def _guarded(branch: ResearchBranch) -> BranchResult:
        async with sem:
            if cancel_event and cancel_event.is_set():
                return BranchResult(
                    branch_id=branch.id,
                    facet=branch.facet,
                    evidence=[],
                    summary="",
                    sources=[],
                    retrieval_count=0,
                    status="failed",
                )
            # Emit parallel start event
            await _emit_live_event(event_emitter, {
                "type": "branch_started",
                "content": {
                    "branch_id": branch.id,
                    "facet": branch.facet,
                    "queries": branch.queries,
                },
                "transient": True,
            })
            try:
                runner = branch_runner or _run_research_branch
                result = await runner(
                    branch=branch,
                    goal=getattr(state, "goal", ""),
                    specs=specs,
                    journal=journal,
                    run_id=run_id,
                    model_override=model_override,
                    event_emitter=event_emitter,
                    cancel_event=cancel_event,
                )
            except Exception as exc:
                logger.exception("runtime.branch_failed", branch_id=branch.id, error=str(exc))
                result = BranchResult(
                    branch_id=branch.id,
                    facet=branch.facet,
                    evidence=[],
                    summary=f"Branch failed: {exc}",
                    sources=[],
                    retrieval_count=0,
                    status="failed",
                )
            await _emit_live_event(event_emitter, {
                "type": "branch_completed",
                "content": {
                    "branch_id": branch.id,
                    "facet": branch.facet,
                    "status": result.status,
                    "retrievals": result.retrieval_count,
                    "sources": len(result.sources),
                },
                "transient": True,
            })
            return result

    # Run all branches concurrently (semaphore limits actual concurrency)
    tasks = [asyncio.ensure_future(_guarded(b)) for b in sorted_branches]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Map back to original branch order
    result_by_id: dict[str, BranchResult] = {}
    for r in results:
        if isinstance(r, BranchResult):
            result_by_id[r.branch_id] = r

    return [result_by_id.get(b.id, BranchResult(
        branch_id=b.id, facet=b.facet, evidence=[],
        summary="", sources=[], retrieval_count=0, status="failed",
    )) for b in branches]


def synthesize_branches(
    state: Any,
    branch_results: list[BranchResult],
    *,
    goal: str = "",
) -> Any:
    """Merge branch results into the run state as a synthesis message."""
    synthesis = merge_branch_results(branch_results)
    state.synthesis = synthesis

    # Add synthesis as a system-level context message
    synthesis_msg = {
        "role": "system",
        "content": (
            f"[Parallel research synthesis]\n"
            f"{synthesis.executive_summary}\n\n"
            f"Total sources: {synthesis.total_sources}\n"
            f"Facets: {list(synthesis.by_facet.keys())}\n"
            f"Evidence items: {len(synthesis.merged_evidence)}\n"
            "Use this synthesis as the factual basis for your final answer."
        ),
    }
    state.messages.append(synthesis_msg)

    # Update retrieval count
    total_retrievals = sum(br.retrieval_count for br in branch_results)
    if hasattr(state, "token_count"):
        state.token_count += estimate_tokens([synthesis_msg])

    return state


async def _run_research_branch_compact(
    branch: ResearchBranch,
    goal: str,
    specs: dict[str, ToolSpec],
    journal: Journal,
    run_id: str,
    *,
    model_override: str | None = None,
    event_emitter: Any = None,
    cancel_event: asyncio.Event | None = None,
    max_steps: int = 30,
    max_tokens: int = 60_000,
    max_cost_usd: float = 2.0,
    max_wall_s: float = 240.0,
) -> BranchResult:
    """Lightweight research branch runner for testing without full worker recursion."""
    # This is a fallback path that does minimal research without spawning a full sub-loop.
    # Used when the full run() recursion is not available (e.g., in tests).
    branch_run_id = f"{run_id}_branch_{branch.id}"
    evidence_items: list[EvidenceItem] = []
    sources: list[str] = []
    retrieval_count = 0

    for query in branch.queries:
        if cancel_event and cancel_event.is_set():
            break

        # Try web_search tool
        search_spec = specs.get("web_search") or specs.get("search") or specs.get("tavily_search")
        if search_spec and search_spec._execute is not None:
            try:
                await j_append_safe(journal, branch_run_id, "tool_call_started",
                                    {"call_id": f"bc_{uuid.uuid4().hex[:8]}", "name": "web_search", "arguments": {"query": query}})
                result = await asyncio.wait_for(search_spec._execute(query=query), timeout=30.0)
                retrieval_count += 1
                urls = extract_urls(result)
                for url in urls:
                    if url not in sources:
                        sources.append(url)
                content = extract_summary(result)
                for url in urls:
                    evidence_items.append(EvidenceItem(
                        url=url, title=f"Search: {query}", content=content, relevance=branch.facet,
                    ))
                await j_append_safe(journal, branch_run_id, "tool_call_result",
                                    {"call_id": f"bc_{uuid.uuid4().hex[:8]}", "name": "web_search", "result": result})

                # Try to open top results
                open_spec = specs.get("open_page") or specs.get("browser_navigate")
                if open_spec and open_spec._execute is not None and urls:
                    top_url = urls[0]
                    try:
                        page_result = await asyncio.wait_for(open_spec._execute(url=top_url), timeout=30.0)
                        page_content = extract_summary(page_result)
                        if page_content:
                            evidence_items.append(EvidenceItem(
                                url=top_url, title=query, content=page_content[:300], relevance=branch.facet,
                            ))
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("runtime.branch_search_failed", branch_id=branch.id, error=str(exc))

    # Generate summary
    summary_parts = [ev.content for ev in evidence_items[:3] if ev.content]
    summary = " ".join(summary_parts)[:500] if summary_parts else f"Collected {len(evidence_items)} evidence items for {branch.facet}."

    status = "completed" if retrieval_count >= branch.min_sources else ("partial" if retrieval_count > 0 else "failed")

    return BranchResult(
        branch_id=branch.id,
        facet=branch.facet,
        evidence=evidence_items,
        summary=summary,
        sources=sources,
        retrieval_count=retrieval_count,
        status=status,  # type: ignore
    )


async def j_append_safe(journal: Journal, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
    """Safe journal append that never raises."""
    try:
        await journal.append(run_id, event_type, payload)
    except Exception:
        pass


async def _run_research_branch_simulated(
    branch: ResearchBranch,
    goal: str,
    specs: dict[str, ToolSpec],
    journal: Journal,
    run_id: str,
    *,
    model_override: str | None = None,
    event_emitter: Any = None,
    cancel_event: asyncio.Event | None = None,
    max_steps: int = 30,
    max_tokens: int = 60_000,
    max_cost_usd: float = 2.0,
    max_wall_s: float = 240.0,
) -> BranchResult:
    """Simulated research branch for environments without search tools."""
    return BranchResult(
        branch_id=branch.id,
        facet=branch.facet,
        evidence=[],
        summary=f"Simulated research for facet: {branch.facet}. Queries: {branch.queries}.",
        sources=[],
        retrieval_count=1,
        status="failed",
    )


async def _run_fast_search_path(
    *,
    state: Any,
    goal: str,
    depth: str,
    specs: dict[str, ToolSpec],
    model_router: Any,
    model_override: str | None,
    event_emitter: Any,
    journal: Any,
    run_id: str,
    cancel_event: Any,
    registry: Any,
    role_config: dict | None,
    user_id: str | None,
    tenant_id: str | None,
    thread_id: str | None,
) -> dict[str, Any]:
    """Single-turn answer for simple factual / comparison queries.

    Bypasses the plan driver, parallel research branches, verifier, and
    memory distillation. Only reachable for top-level runs (not sub-branches)
    whose research depth is "none" (direct answer) or "quick" (one web search
    then answer). Intended for short goals like "Manus vs Tenssa" or "what is
    the capital of France".
    """
    g = goal or getattr(state, "goal", "") or ""
    short_goal = len(g.split()) <= 20

    # Build a minimal message history for the model.
    messages: list[dict[str, Any]] = []
    system_prompt = (
        "Answer the user's question directly and concisely. "
        "If you need current facts, call web_search first, then synthesize "
        "with citations. Do not over-explain."
    )
    messages.append({"role": "system", "content": system_prompt})
    if getattr(state, "messages", None):
        messages.extend(state.messages)
    messages.append({"role": "user", "content": g})

    # Select tools: web_search ("browser") for quick depth, none otherwise.
    search_tool_name: str | None = None
    tool_schemas: list[dict[str, Any]] = []
    if depth == "quick" and short_goal and registry is not None:
        search_tool = registry.get("browser") or registry.get("web_search")
        if search_tool is not None:
            tool_schemas = [search_tool.schema.to_dict()]
            search_tool_name = search_tool.schema.name

    # First model call: answer directly, or decide to search.
    response = await model_complete(
        messages,
        tools=tool_schemas,
        role="fast_path",
        tool_choice="auto" if tool_schemas else "none",
        model_override=model_override,
        goal=g,
        research_depth=depth,
        tools_used=sorted([search_tool_name] if search_tool_name else []),
    )

    # If the model asked to search, execute the tool directly and re-call.
    if response.tool_calls and search_tool_name:
        search_tool = registry.get("browser") or registry.get("web_search")
        if search_tool is not None:
            for tc in response.tool_calls:
                if tc.get("name") != search_tool_name:
                    continue
                args = tc.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                try:
                    search_result = await search_tool.execute(**args)
                except Exception as exc:  # noqa: BLE001
                    search_result = {"output": f"Search failed: {exc}"}
                call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": response.tool_calls,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(search_result),
                    }
                )
                break

            # Second model call: synthesize the final answer from search results.
            response = await model_complete(
                messages,
                tools=[],
                role="fast_path",
                tool_choice="none",
                model_override=model_override,
                goal=g,
                research_depth=depth,
                tools_used=[],
            )

    finish_payload = {
        "summary": response.content or "I couldn't find an answer.",
        "status": "completed",
        "evidence_refs": [],
    }
    await journal.append(run_id, "run_finished", finish_payload)
    return _terminal_return(
        "completed",
        finish_payload,
        state,
        run_id=run_id,
    )


async def run(
    run_id: str,
    goal: str,
    role_config: dict | None = None,
    *,
    registry: Any = None,
    tool_registry: Any = None,
    session_factory: Any = None,
    workspace: str | Path | None = None,
    journal: Journal | None = None,
    event_emitter: Any = None,
    model_override: str | None = None,
    model_router: Any = None,
    context_budget_chars: int | None = None,
    parent_run_id: str | None = None,
    system_prompt: str | None = None,
    tool_allowlist: tuple | list | None = None,
    max_steps: int | None = None,
    max_tokens: int | None = None,
    max_cost_usd: float | None = None,
    max_wall_s: float | None = None,
    cancel_event: Any = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
    thread_id: str | None = None,
    confirmed_actions: dict[str, Any] | None = None,
    rejected_tool_call: dict[str, Any] | None = None,
    schedule_authorization: dict[str, Any] | None = None,
    **_extra: Any,
) -> dict[str, Any]:
    role_config = dict(role_config or {})
    if parent_run_id:
        role_config["parent_run_id"] = parent_run_id
    if system_prompt:
        role_config["system_prompt"] = system_prompt
    if tool_allowlist is not None:
        role_config["tool_allowlist"] = list(tool_allowlist)
    if user_id:
        role_config["user_id"] = user_id
    if tenant_id:
        role_config["tenant_id"] = tenant_id
        role_config["org_id"] = tenant_id
    if thread_id:
        role_config["thread_id"] = thread_id
    # Scheduled-task autonomy: tools the operator explicitly delegated
    # through the schedule's autonomy policy (additional_allowed_tools /
    # allowed_custom_mcp_tools) run without an interactive confirmation so
    # unattended executions do not deadlock on an approval no one can
    # resolve. The argument-provenance guard still applies, so a delegated
    # send/spend tool whose target value looks injected still requires
    # approval and safely pauses the unattended run.
    org_policy = org_policy_from_schedule(schedule_authorization)
    if registry is None and tool_registry is not None:
        registry = tool_registry
    if model_router is not None:
        try:
            from app.runtime.models import set_model_router
            set_model_router(model_router)
        except Exception:
            pass
    j = journal or default_journal
    if session_factory is not None:
        j.set_session_factory(session_factory)

    ws = Path(
        workspace
        or os.environ.get("AGENT_WORKSPACE")
        or f"/tmp/onsite48_workspace/{run_id}"
    )
    ws.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AGENT_WORKSPACE", str(ws))
    os.environ["AGENT_RUN_ID"] = run_id

    # Persistent computer for this run
    try:
        from app.runtime.workspace import create_workspace

        computer = await create_workspace(run_id)
        snap = await computer.snapshot()
        logger.info("runtime.workspace.ready", run_id=run_id, snapshot=snap)
    except Exception as exc:
        logger.warning("runtime.workspace.create_failed", run_id=run_id, error=str(exc))
        computer = None

    state = await j.replay(run_id, goal=goal, role_config=role_config)
    # Follow-up context: if this is a follow-up and we have conversation history,
    # prepend it to messages so the model knows what "them" refers to.
    if not state.messages:
        conv_ctx = role_config.get("conversation_context") if role_config else ""
        is_follow = role_config.get("is_follow_up") if role_config else False
        if is_follow and conv_ctx:
            # conversation_context is a formatted transcript: "User: ...\nSynth: ..."
            for line in conv_ctx.split("\n"):
                if line.startswith("User: "):
                    state.messages.append({"role": "user", "content": line[6:]})
                elif line.startswith("Synth: "):
                    state.messages.append({"role": "assistant", "content": line[7:]})
    forced_approved_call: dict[str, Any] | None = None
    if not state.goal:
        state.goal = goal
    if state.status in ("finished", "failed") and state.finish_payload:
        pending = state.finish_payload.get("pending_call") if isinstance(state.finish_payload, dict) else None
        approval_wait = isinstance(pending, dict) and state.finish_payload.get("status") == "waiting_approval"
        rejected = rejected_tool_call or {}
        if approval_wait and rejected.get("tool_name") == pending.get("name") and rejected.get("tool_args") == pending.get("arguments"):
            declined = {
                "summary": f"You declined the requested {pending.get('name')} action. No website was published.",
                "status": "cancelled",
                "evidence_refs": [],
            }
            await j.append(run_id, "run_finished", declined)
            return _terminal_return("cancelled", declined, state, run_id=run_id)
        was_confirmed = False
        if approval_wait and pending:
            from app.agents.legacy_compat import confirmation_key
            approval_key = confirmation_key(
                str(role_config.get("thread_id") or thread_id or run_id),
                int(state.step_count or 0),
                str(pending.get("name") or ""),
                dict(pending.get("arguments") or {}),
            )
            was_confirmed = bool((confirmed_actions or {}).get(approval_key))
        if approval_wait and was_confirmed:
            # Re-submit the exact paused call without asking the model to restate
            # or widen the user's approved arguments.
            forced_approved_call = {
                "id": pending.get("call_id") or f"approved_{uuid.uuid4().hex[:8]}",
                "name": pending.get("name"),
                "arguments": dict(pending.get("arguments") or {}),
            }
            state.status = "running"
            state.finish_payload = None
        else:
            return _terminal_return(
                state.status, state.finish_payload, state, run_id=run_id
            )
    approved_tool_calls = dict(confirmed_actions or {})

    specs = build_tool_specs(registry, workspace_root=str(ws))
    # Filter tools by allowlist if present (workers inherit subset)
    _allow = (role_config or {}).get("tool_allowlist")
    if _allow is not None:
        allow_set = set(_allow) | {"finish", "ask_user"}
        specs = {k: v for k, v in specs.items() if k in allow_set}
    context = ContextBuilder(
        budget_chars=context_budget_chars or int(
            os.environ.get("AGENT_CONTEXT_BUDGET", "80000")
        ),
        workspace=ws,
        journal=j,
    )

    # Phase 1: Context window manager for sliding-window summarization
    context_manager = ContextManager(
        max_tokens=DEFAULT_MAX_TOKENS_PHASE1,
        summary_threshold=DEFAULT_SUMMARY_THRESHOLD,
        summarizer_model="fast",
    )

    _rb = role_config.get("budget") or {}

    # Capture caller-explicitness BEFORE resolution.
    _explicit = {
        "steps": max_steps is not None,
        "tokens": max_tokens is not None,
        "cost": max_cost_usd is not None,
        "wall": max_wall_s is not None,
    }
    max_steps = int(
        max_steps if _explicit["steps"] else role_config.get("max_steps") or _rb.get("steps") or DEFAULT_MAX_STEPS
    )
    max_tokens = int(
        max_tokens if _explicit["tokens"] else role_config.get("max_tokens") or _rb.get("tokens") or DEFAULT_MAX_TOKENS
    )
    max_cost = float(
        max_cost_usd if _explicit["cost"] else role_config.get("max_cost_usd") or _rb.get("cost_usd") or DEFAULT_MAX_COST_USD
    )
    max_wall = float(
        max_wall_s if _explicit["wall"] else role_config.get("max_wall_s") or _rb.get("wall_s") or DEFAULT_MAX_WALL_S
    )

    # Adaptive scaling for complex multi-deliverable goals only.
    # Pure "research X" no longer doubles budgets; depth caps apply below.
    _goal_for_budget = goal or getattr(state, "goal", "") or ""
    _mult = complexity_multiplier(_goal_for_budget)
    if _mult > 1.0 and not _explicit["steps"] and not role_config.get("max_steps") and not _rb.get("steps"):
        max_steps = min(int(max_steps * _mult), DEFAULT_MAX_STEPS * 2)
        max_wall = min(max_wall * _mult, DEFAULT_MAX_WALL_S * 2    )

    # Research depth mode: quick | standard | deep caps (P1 product control).
    _explicit_depth = None
    if isinstance(role_config, dict):
        _explicit_depth = (
            role_config.get("research_depth")
            or (role_config.get("budget") or {}).get("research_depth")
            or role_config.get("researchDepth")
        )
        if isinstance(role_config.get("budget"), dict) and not _explicit_depth:
            _explicit_depth = role_config["budget"].get("research_depth") or role_config["budget"].get("researchDepth")
    max_steps, max_wall, _research_plan = apply_depth_budget_caps(
        max_steps=max_steps,
        max_wall=max_wall,
        goal=_goal_for_budget,
        explicit_depth=_explicit_depth,
        explicit_steps=_explicit["steps"] or bool(role_config.get("max_steps") or _rb.get("steps")),
        explicit_wall=_explicit["wall"] or bool(role_config.get("max_wall_s") or _rb.get("wall_s")),
    )

    # Worker budget floor: ensure workers (parent_run_id set) never receive less
    # than the research minimums unless explicitly configured.
    if parent_run_id and not _explicit["steps"] and max_steps < 30:
        max_steps = 30
    if parent_run_id and not _explicit["wall"] and max_wall < 600:
        max_wall = 600.0

    # ─── Fast path: simple factual / comparison queries ──────────────────
    # Short-circuit before the plan driver, parallel branches, verifier, and
    # distillation for top-level runs that need at most one web search.
    # Exclude goals that need browser/computer interaction (click, navigate,
    # open a page, etc.) — those go through the normal tool loop.
    _g_for_depth = goal or getattr(state, "goal", "") or ""
    _browser_signals = any(
        k in _g_for_depth.lower()
        for k in ("click", "website", "web page", "browser", "screenshot", "login to", "navigate", "open an example")
    )
    if (
        not parent_run_id
        and not _browser_signals
        and _research_plan.depth in ("none", "quick")
        and (_research_plan.depth == "none" or len(_g_for_depth.split()) <= 20)
    ):
        return await _run_fast_search_path(
            state=state,
            goal=goal,
            depth=_research_plan.depth,
            specs=specs,
            model_router=model_router,
            model_override=model_override,
            event_emitter=event_emitter,
            journal=j,
            run_id=run_id,
            cancel_event=cancel_event,
            registry=registry,
            role_config=role_config,
            user_id=user_id,
            tenant_id=tenant_id,
            thread_id=thread_id,
        )

    # Fail fast when the goal looks like research but no search/browse tool exists.
    # Skipped when the fast path already returned (depth="none" direct answers
    # need no search tools; depth="quick" may use the registered browser tool).
    _goal_l = (goal or getattr(state, "goal", None) or "").lower()
    _researchy = is_research_goal(goal or getattr(state, "goal", None) or "")
    _search_tools = {
        n for n in specs
        if any(x in n.lower() for x in ("search", "web_", "browser_", "brave", "tavily", "http_get", "fetch"))
    }
    if _researchy and not _search_tools and not parent_run_id:
        finish_payload = {
            "summary": (
                "Web research tools are not available in this environment "
                "(search/MCP not configured). Configure a search provider or "
                "browser bridge, then retry."
            ),
            "status": "failed",
            "evidence_refs": [],
            "error_code": "search_unavailable",
            "retryable": True,
            "suggestion": "Enable web search or pair My Browser, then run the task again.",
        }
        await j.append(run_id, "run_failed", finish_payload)
        return _terminal_return("failed", finish_payload, state, run_id=run_id, specs=specs)

    started = time.monotonic()
    import asyncio as _asyncio
    cancel_event = cancel_event or _asyncio.Event()

    _cancel_thread_id = thread_id or (role_config or {}).get("thread_id")
    _registered_cancel = bool(_cancel_thread_id)
    if _registered_cancel:
        try:
            from app.services.task_execution_control import task_execution_control
            await task_execution_control.register_cancel_event(_cancel_thread_id, cancel_event)
        except Exception:
            _registered_cancel = False

    async def _unregister_cancel_event() -> None:
        if _registered_cancel and _cancel_thread_id:
            try:
                from app.services.task_execution_control import task_execution_control
                await task_execution_control.unregister_cancel_event(_cancel_thread_id)
            except Exception:
                pass

    from app.runtime.roles_v2 import RoleConfig as _RC, get_role as _get_role, LEAD as _LEAD
    from app.runtime.locks import ResourceBusy, get_lock_manager, resources_for_tool
    from app.runtime.spawn import SpawnManager

    try:
        _role_name = role_config.get("name") or ("lead" if not parent_run_id else "researcher")
        if _role_name in ("lead", "researcher", "browser_operator", "coder", "writer", "critic"):
            _role = _get_role(_role_name)
        else:
            _role = _LEAD
        overrides = {k: role_config[k] for k in ("tool_allowlist", "system_prompt_file", "budget", "may_spawn", "depth_cap") if k in role_config}
        if overrides:
            _role = _RC.from_dict({**_role.to_dict(), **overrides})
    except Exception:
        _role = _LEAD

    spawn_mgr = SpawnManager(
        parent_run_id=run_id,
        parent_role=_role,
        workspace_root=ws,
        loop_runner=None,
        journal=j,
        depth=1 if parent_run_id else 0,
        cancel_event=cancel_event,
    )

    async def _child_loop_runner(**kwargs):
        return await run(**kwargs)

    spawn_mgr.loop_runner = _child_loop_runner

    # Loop-detection bookkeeping
    recent_calls: list[tuple[str, str]] = []  # (name, args_hash)
    same_call_streak = 0
    no_progress_streak = 0
    last_progress_sig: str | None = None
    empty_response_nudges = 0
    finish_attempts = 0
    research_saturation_nudged = False

    auth_refresh_attempted: set[str] = set()
    untrusted_values: list[str] = list(getattr(state, "untrusted_values", None) or [])
    readbacks_ledger: list[dict[str, Any]] = []
    key_results: list[dict[str, Any]] = []
    _retrieval_count = weighted_retrieval_count(key_results)
    if _research_plan.is_research:
        await j.append(
            run_id,
            "research_plan",
            {
                "depth": _research_plan.depth,
                "label": _research_plan.label,
                "min_retrievals_before_finish": _research_plan.min_retrievals_before_finish,
                "saturation_retrievals": _research_plan.saturation_retrievals,
                "max_steps": max_steps,
                "max_wall_s": max_wall,
                "branch_count": len(_research_plan.branches),
            },
        )

    # Phase 1: Parallel research branch execution — runs once at loop entry
    _parallel_branches_done = False

    # Phase 2 adoption: hierarchical plan awareness (fail-safe, env-gated).
    _hier_plan_active = False
    if ENABLE_HIERARCHICAL_PLANS and not parent_run_id and _explicit_depth != "quick":
        from app.runtime.plan_driver import maybe_create_plan
        _hier_plan_active = await maybe_create_plan(
            run_id=run_id, goal=state.goal or goal or "", specs=specs, journal=j,
            event_emitter=event_emitter, model_override=model_override, model_router=model_router,
        )

    # Live plan tracking: every executed tool call becomes one step the
    # client watches appear and fill in as the run progresses.
    live_plan_steps: list[dict[str, Any]] = []
    tool_step_counter = 0

    if not state.messages:
        await j.append(run_id, "run_started", {"goal": goal, "role_config": role_config})
        if _explicit_depth != "quick":
            try:
                from app.runtime.memory_distiller import get_memory_store
                _store = get_memory_store()
                _lessons = await _store.recent(limit=5, org_id=str((role_config or {}).get("org_id") or tenant_id or "default")) if hasattr(_store, "recent") else []
                _block = _lessons_context_block(_lessons)
                if _block:
                    state.messages.append({"role": "system", "content": _block})
            except Exception as exc:
                logger.debug("loop.lessons_inject_failed", error=str(exc)[:150])

    logger.info("run.start", run_id=run_id, goal=(goal or "")[:120])

    while True:
        # Budget gates
        if cancel_event.is_set():
            await spawn_mgr.cancel_all()
            finish_payload = {
                "summary": "Run cancelled (parent or external cancel).",
                "status": "cancelled",
                "evidence_refs": [],
            }
            await j.append(run_id, "run_finished", finish_payload)
            await _unregister_cancel_event()
            return _terminal_return("cancelled", finish_payload, state, run_id=run_id, specs=specs)

        if state.step_count >= max_steps:
            finish_payload = {
                "summary": "Stopped: step budget exhausted",
                "status": "partial",
                "evidence_refs": [],
            }
            await j.append(run_id, "run_finished", finish_payload)
            return _terminal_return("partial", finish_payload, state, run_id=run_id, specs=specs)
        if state.token_count >= max_tokens:
            finish_payload = {
                "summary": "Stopped: token budget exhausted",
                "status": "partial",
                "evidence_refs": [],
            }
            await j.append(run_id, "run_finished", finish_payload)
            return _terminal_return("partial", finish_payload, state, run_id=run_id, specs=specs)
        if state.cost_usd >= max_cost:
            finish_payload = {
                "summary": "Stopped: cost budget exhausted",
                "status": "partial",
                "evidence_refs": [],
            }
            await j.append(run_id, "run_finished", finish_payload)
            return _terminal_return("partial", finish_payload, state, run_id=run_id, specs=specs)
        if (time.monotonic() - started) >= max_wall:
            finish_payload = {
                "summary": "Stopped: wall-time budget exhausted",
                "status": "partial",
                "evidence_refs": [],
            }
            await j.append(run_id, "run_finished", finish_payload)
            return _terminal_return("partial", finish_payload, state, run_id=run_id, specs=specs)

        # Phase 1: Parallel research branch execution (runs once per run)
        if (
            not _parallel_branches_done
            and ENABLE_PARALLEL_RESEARCH
            and _research_plan.is_research
            and _research_plan.branches
            and not parent_run_id  # only at the lead level
        ):
            _parallel_branches_done = True
            _branches = _research_plan.branches
            logger.info(
                "runtime.parallel_branches_entering",
                run_id=run_id,
                branch_count=len(_branches),
                facets=[b.facet for b in _branches],
            )
            await _emit_live_event(event_emitter, {
                "type": "parallel_branches_started",
                "content": {
                    "branch_count": len(_branches),
                    "branches": [{"id": b.id, "facet": b.facet, "priority": b.priority} for b in _branches],
                },
                "transient": True,
            })
            try:
                branch_results = await run_research_branches_parallel(
                    state=state,
                    branches=_branches,
                    specs=specs,
                    journal=j,
                    run_id=run_id,
                    model_override=model_override,
                    event_emitter=event_emitter,
                    cancel_event=cancel_event,
                )
                for br in branch_results:
                    await j.append(
                        run_id,
                        "branch_result",
                        {
                            "branch_id": br.branch_id,
                            "facet": br.facet,
                            "status": br.status,
                            "retrieval_count": br.retrieval_count,
                            "sources": br.sources,
                            "summary": br.summary,
                        },
                    )
                state = synthesize_branches(state, branch_results, goal=goal or getattr(state, "goal", ""))
                await j.append(
                    run_id,
                    "synthesis_result",
                    {
                        "executive_summary": getattr(state, "synthesis", None) and state.synthesis.executive_summary,
                        "total_sources": getattr(state, "synthesis", None) and state.synthesis.total_sources,
                        "facets": list(getattr(state, "synthesis", None).by_facet.keys()) if getattr(state, "synthesis", None) else [],
                    },
                )
                # Add branch retrievals to the count
                for br in branch_results:
                    _retrieval_count += br.retrieval_count
                    # Add branch evidence to key_results for verifier
                    for ev in br.evidence:
                        key_results.append({
                            "tool": "web_search",
                            "call_id": f"branch_{br.branch_id}",
                            "result": {"url": ev.url, "title": ev.title, "content": ev.content[:500]},
                            "result_summary": ev.content[:300],
                            "evidence_ref": ev.url,
                        })
                await j.append(
                    run_id,
                    "nudge",
                    {"reason": "parallel_research_complete", "branches": len(_branches)},
                )
                # Emit progress with parallel retrieval count
                await _emit_live_event(event_emitter, {
                    "type": "run_progress", "transient": True,
                    "content": {
                        "retrievals": _retrieval_count,
                        "steps": state.step_count,
                        "depth": _research_plan.depth,
                        "finish_attempts": finish_attempts,
                        "parallel_branches": len(_branches),
                    },
                })
                continue  # Skip normal single-branch flow; model sees synthesis
            except Exception as exc:
                logger.exception("runtime.parallel_branches_failed", run_id=run_id, error=str(exc))
                await _emit_live_event(event_emitter, {
                    "type": "parallel_branches_failed",
                    "content": {"error": str(exc)},
                    "transient": True,
                })
                # Fall through to normal single-branch flow

        messages = context.build(state, specs)
        tool_schemas = specs_for_model(specs)

        if _hier_plan_active:
            from app.runtime.plan_driver import plan_context_message
            _plan_msg = plan_context_message(run_id)
            if _plan_msg is not None:
                messages = [_plan_msg] + messages

        try:
            if forced_approved_call is not None:
                from types import SimpleNamespace
                forced_call = forced_approved_call
                forced_approved_call = None
                response = SimpleNamespace(
                    content="",
                    tool_calls=[forced_call],
                    usage={},
                    cost_usd=0.0,
                    path="explicit-approval-replay",
                    model="runtime",
                )
            else:
                call_task = asyncio.ensure_future(
                    model_complete(
                        messages,
                        tools=tool_schemas,
                        role="loop",
                        tool_choice="auto",
                        model_override=model_override,
                        goal=state.goal or goal,
                        research_depth=_explicit_depth,
                        tools_used=sorted(specs.keys()),
                    )
                )
                cancel_watcher = asyncio.ensure_future(cancel_event.wait())
                try:
                    done, pending = await asyncio.wait(
                        {call_task, cancel_watcher},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if cancel_event.is_set() and call_task not in done:
                        call_task.cancel()
                        try:
                            await asyncio.wait_for(call_task, timeout=5.0)
                        except Exception:
                            pass
                        finish_payload = {
                            "summary": "Run cancelled by user.",
                            "status": "cancelled",
                            "evidence_refs": [],
                        }
                        await j.append(run_id, "run_finished", finish_payload)
                        await _unregister_cancel_event()
                        return _terminal_return("cancelled", finish_payload, state, run_id=run_id, specs=specs)
                    response = await call_task
                finally:
                    cancel_watcher.cancel()
        except asyncio.CancelledError:
            finish_payload = {
                "summary": "Run cancelled by user.",
                "status": "cancelled",
                "evidence_refs": [],
            }
            await j.append(run_id, "run_finished", finish_payload)
            await _unregister_cancel_event()
            return _terminal_return("cancelled", finish_payload, state, run_id=run_id, specs=specs)
        except Exception as exc:
            logger.exception("runtime.loop.model_error", error=str(exc))
            finish_payload = {
                "summary": f"Model error: {exc}",
                "status": "failed",
                "evidence_refs": [],
            }
            await j.append(run_id, "run_failed", finish_payload)
            return _terminal_return("failed", finish_payload, state, run_id=run_id, specs=specs)

        await j.append(
            run_id,
            "model_response",
            {
                "tokens": (response.usage or {}).get("input_tokens", 0)
                + (response.usage or {}).get("output_tokens", 0),
                "cost_usd": response.cost_usd,
                "path": response.path,
                "model": response.model,
                "message": {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    or None,
                },
            },
        )
        state.messages.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": response.tool_calls or None,
            }
        )
        state.step_count += 1
        state.token_count += int((response.usage or {}).get("input_tokens", 0)) + int(
            (response.usage or {}).get("output_tokens", 0)
        )
        state.cost_usd += float(response.cost_usd or 0)

        # Live cost visibility: every model call updates the header
        # cost pill while the task is still running.
        await _emit_live_event(event_emitter, {
            "type": "cost_update",
            "cost_metrics": {"total_cost_usd": round(float(state.cost_usd or 0), 6)},
            "transient": True,
        })
        await _emit_live_event(event_emitter, {
            "type": "run_progress", "transient": True,
            "content": {
                "retrievals": _retrieval_count,
                "steps": state.step_count,
                "depth": getattr(locals().get("_research_plan"), "depth", None),
                "finish_attempts": finish_attempts,
            },
        })

        # Phase 1: Context window management — summarize after each model turn
        if ENABLE_CONTEXT_SUMMARIZATION:
            state = await context_manager.maybe_summarize(state)
            if getattr(state, "summarized_turns", 0) > 0:
                await j.append(
                    run_id,
                    "context_summarized",
                    {
                        "summarized_turns": state.summarized_turns,
                        "token_count": state.token_count,
                        "max_tokens": context_manager.max_tokens,
                    },
                )

        if not response.tool_calls:
            empty_response_nudges += 1
            _goal_now = state.goal or goal or ""
            _research_now = is_research_goal(_goal_now)
            _retrievals = _retrieval_count
            if empty_response_nudges == 1:
                if _research_now and _retrievals < (_research_plan.min_retrievals_before_finish or 1):
                    state.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "This is a research goal. Do not answer from memory alone. "
                                "Call web_search or open_page to gather sources, then synthesize and finish "
                                "with evidence_refs. A prose-only answer without retrieval is not acceptable."
                            ),
                        }
                    )
                    await j.append(run_id, "nudge", {"reason": "research_requires_retrieval", "retrievals": _retrievals})
                    continue
                if response.content:
                    state.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "If the goal is complete, call the finish tool with summary and status. "
                                "Otherwise call a tool or ask_user."
                            ),
                        }
                    )
                    await j.append(run_id, "nudge", {"reason": "no_tool_calls"})
                    continue
            # Second+ empty tool-call turn
            if _research_now and _retrievals < (_research_plan.min_retrievals_before_finish or 1):
                finish_payload = {
                    "summary": (
                        "Research did not run: the model produced an answer without using "
                        "search or page tools. Please send the request again, or rephrase "
                        "as an explicit research task."
                    ),
                    "status": "failed",
                    "evidence_refs": [],
                    "error_code": "research_no_retrieval",
                    "retryable": True,
                    "stop_reason": "no_tool_calls_research",
                }
                await j.append(run_id, "run_finished", finish_payload)
                return _terminal_return("failed", finish_payload, state, run_id=run_id, specs=specs)
            finish_payload = {
                "summary": response.content or "Agent stopped without calling finish.",
                "status": "partial",
                "evidence_refs": [],
                "stop_reason": "no_tool_calls",
            }
            await j.append(run_id, "run_finished", finish_payload)
            return _terminal_return("partial", finish_payload, state, run_id=run_id, specs=specs)

        empty_response_nudges = 0

        for tc in response.tool_calls:
            name = tc.get("name") or ""
            args = tc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            spec = specs.get(name)

            await j.append(
                run_id,
                "tool_call_proposed",
                {"call_id": call_id, "name": name, "arguments": args},
            )

            if spec is None:
                result = _structured_error(
                    "unknown_tool", f"Tool {name!r} is not registered"
                )
                msg = context.format_tool_result(
                    call_id=call_id,
                    name=name,
                    result=result,
                    spec=ToolSpec(
                        name=name,
                        description="",
                        parameters={},
                        side_effect="read",
                        idempotency="idempotent",
                        timeout_s=5,
                        max_output_chars=2000,
                        trust="trusted",
                    ),
                )
                state.messages.append(msg)
                await j.append(
                    run_id,
                    "tool_call_result",
                    {"call_id": call_id, "name": name, "result": result, "message": msg},
                )
                continue

            # ----- Policy (with provenance) -----
            run_state_dict = state.to_dict()
            run_state_dict["untrusted_values"] = untrusted_values
            run_state_dict["approved_contacts"] = (
                role_config.get("approved_contacts") or []
            )
            from app.agents.legacy_compat import confirmation_key
            approval_key = confirmation_key(
                str(role_config.get("thread_id") or thread_id or run_id),
                int(state.step_count or 0),
                name,
                args,
            )
            if approved_tool_calls.pop(approval_key, False):
                decision = PolicyDecisionResult(
                    Decision.ALLOW,
                    "the user approved this exact tool action",
                )
            else:
                decision = policy_evaluate(
                    {"name": name, "arguments": args},
                    spec,
                    run_state_dict,
                    org_policy=org_policy,
                )
            await j.append(
                run_id,
                "policy_decision",
                {
                    "call_id": call_id,
                    "name": name,
                    "decision": decision.decision.value,
                    "reason": decision.reason,
                    "metadata": decision.metadata,
                },
            )

            if decision.decision == Decision.DENY:
                result = _structured_error(
                    "policy_denied",
                    decision.reason,
                    retryable=False,
                    suggestion="Choose a different tool or ask_user",
                )
                msg = context.format_tool_result(
                    call_id=call_id, name=name, result=result, spec=spec
                )
                state.messages.append(msg)
                await j.append(
                    run_id,
                    "tool_call_result",
                    {"call_id": call_id, "name": name, "result": result, "message": msg},
                )
                continue

            if decision.decision in (Decision.APPROVAL, Decision.TAKEOVER):
                await j.append(
                    run_id,
                    "approval_requested",
                    {
                        "call_id": call_id,
                        "name": name,
                        "arguments": args,
                        "reason": decision.reason,
                        "metadata": decision.metadata,
                    },
                )
                finish_payload = {
                    "summary": (
                        f"Approval required for {name}: {decision.reason}. "
                        f"Provenance: {json.dumps(decision.metadata.get('provenance', []), default=str)[:500]}"
                    ),
                    "status": "waiting_approval",
                    "evidence_refs": [],
                    "pending_call": {"name": name, "arguments": args, "call_id": call_id},
                }
                await j.append(run_id, "run_finished", finish_payload)
                return _terminal_return(
                    "waiting_approval", finish_payload, state, run_id=run_id, specs=specs
                )

            # ----- Phase 4: Learning from Corrections (Qdrant-backed) -----
            # Before executing, probe the correction store for semantically
            # similar past corrections and surface them as a hint in the journal.
            # This is non-blocking: memory unavailability degrades gracefully.
            try:
                from app.services.correction_store import correction_store
                _org_id = (role_config or {}).get("org_id") or tenant_id or "default"
                _user_id = role_config.get("user_id") or user_id or "default"
                _thread_id = str(role_config.get("thread_id") or thread_id or run_id)
                # Only probe for tools that have meaningful arguments and are not loop/control tools
                if name not in ("finish", "ask_user", "reset_prompt", "request_user_auth") and args:
                    _suggestion = await correction_store.suggest_correction(
                        organization_id=str(_org_id),
                        user_id=str(_user_id),
                        tool_name=name,
                        tool_args=args if isinstance(args, dict) else {"value": str(args)},
                        goal=state.goal or goal,
                        thread_id=_thread_id if thread_id else None,
                    )
                    if _suggestion is not None:
                        await j_append_safe(
                            j, run_id, "correction_suggestion", {
                                "call_id": call_id,
                                "name": name,
                                "score": _suggestion.get("score", 0.0),
                                "correction_type": _suggestion.get("correction_type", ""),
                                "corrected_args": _suggestion.get("corrected_args", {}),
                                "user_feedback": _suggestion.get("user_feedback", ""),
                            }
                        )
                        # Surface as a system hint in the conversation
                        _hint = _suggestion.get("user_feedback") or _suggestion.get("corrected_args", {})
                        if _hint:
                            state.messages.append({
                                "role": "user",
                                "content": (
                                    f"[Learning hint from past correction] "
                                    f"A similar {name} call was corrected before. "
                                    f"Suggestion: {_hint if isinstance(_hint, str) else json.dumps(_hint)[:500]}"
                                ),
                            })
            except Exception as _cs_exc:
                logger.debug("loop.correction_probe_skipped", error=str(_cs_exc)[:120])

            # ----- spawn / wide_map (Phase 5) -----
            if name in ("spawn_agent", "spawn_agents", "await_agents", "wide_map"):
                await j.append(
                    run_id,
                    "tool_call_started",
                    {"call_id": call_id, "name": name, "arguments": args},
                )
                try:
                    if name == "spawn_agent":
                        handle = await spawn_mgr.spawn_agent(
                            role=args.get("role") or "researcher",
                            goal=args.get("goal") or "",
                            inputs=args.get("inputs") or [],
                            output_schema=args.get("output_schema"),
                            budget=args.get("budget"),
                        )
                        result = {
                            "worker_id": handle.worker_id,
                            "run_id": handle.run_id,
                            "role": handle.role,
                            "status": handle.status,
                            "parent_run_id": run_id,
                        }
                    elif name == "spawn_agents":
                        handles = await spawn_mgr.spawn_agents(args.get("specs") or [])
                        result = {
                            "handles": [
                                {
                                    "worker_id": h.worker_id,
                                    "run_id": h.run_id,
                                    "role": h.role,
                                    "status": h.status,
                                }
                                for h in handles
                            ]
                        }
                    elif name == "await_agents":
                        wids = args.get("worker_ids")
                        handles = list(spawn_mgr._handles.values())
                        if wids:
                            handles = [h for h in handles if h.worker_id in wids]
                        handles = await spawn_mgr.await_agents(
                            handles, timeout_s=args.get("timeout_s")
                        )
                        child_cost = spawn_mgr.total_child_cost()
                        state.cost_usd = float(state.cost_usd or 0) + child_cost
                        result = {
                            "workers": [
                                {
                                    "worker_id": h.worker_id,
                                    "run_id": h.run_id,
                                    "status": h.status,
                                    "summary": h.summary,
                                    "artifacts": h.artifacts,
                                    "cost_usd": h.cost_usd,
                                    "error": h.error,
                                }
                                for h in handles
                            ],
                            "total_child_cost_usd": child_cost,
                        }
                    else:
                        from app.runtime.spawn import wide_map as _wide_map
                        result = await _wide_map(
                            items=args.get("items") or [],
                            role=args.get("role") or "researcher",
                            instruction_template=args.get("instruction_template") or "{item}",
                            output_schema=args.get("output_schema"),
                            concurrency=int(args.get("concurrency") or 4),
                            per_item_budget=args.get("per_item_budget"),
                            parent_run_id=run_id,
                            parent_role=_role,
                            workspace_root=ws,
                            loop_runner=spawn_mgr.loop_runner,
                            journal=j,
                            depth=spawn_mgr.depth,
                            cancel_event=cancel_event,
                            reduce_role=args.get("reduce_role"),
                            reduce_instruction=args.get("reduce_instruction"),
                        )
                        state.cost_usd = float(state.cost_usd or 0) + float(
                            result.get("cost_usd") or 0
                        )
                except Exception as exc:
                    result = _structured_error(
                        "spawn_error",
                        f"{type(exc).__name__}: {exc}",
                        retryable=False,
                    )
                msg = context.format_tool_result(
                    call_id=call_id, name=name, result=result, spec=spec
                )
                state.messages.append(msg)
                await j.append(
                    run_id,
                    "tool_call_result",
                    {"call_id": call_id, "name": name, "result": result, "message": msg},
                )
                key_results.append({"tool": name, "result": result})
                _retrieval_count += weighted_retrieval_increment(name, result)
                continue

            # ----- resource locks for mutating tools -----
            lock_keys = resources_for_tool(name, args if isinstance(args, dict) else {})
            lock_holder = None
            lock_mgr = get_lock_manager()
            if lock_keys and getattr(spec, "side_effect", "read") != "read":
                try:
                    for lk in lock_keys:
                        lock_holder = await lock_mgr.acquire(
                            lk, run_id=run_id, wait_s=10.0
                        )
                except ResourceBusy as rb:
                    result = rb.to_dict()
                    msg = context.format_tool_result(
                        call_id=call_id, name=name, result=result, spec=spec
                    )
                    state.messages.append(msg)
                    await j.append(
                        run_id,
                        "tool_call_result",
                        {"call_id": call_id, "name": name, "result": result, "message": msg},
                    )
                    continue

            # ----- finish / ask_user special cases -----
            if name == "finish":
                finish_attempts += 1
                finish_status = args.get("status") or "completed"
                # normalize legacy needs_user
                if finish_status == "needs_user":
                    finish_status = "needs_input"
                finish_summary = args.get("summary") or ""
                finish_payload = {
                    "summary": finish_summary,
                    "status": finish_status,
                    "evidence_refs": args.get("evidence_refs") or [],
                }
                await j.append(
                    run_id,
                    "tool_call_started",
                    {"call_id": call_id, "name": name, "arguments": args},
                )

                # Research evidence gate: block premature success finishes.
                if finish_status in ("completed", "completed_unverified", "partial"):
                    _block = research_finish_blocked(
                        state.goal or goal or "",
                        key_results,
                        explicit_depth=(role_config or {}).get("research_depth")
                        if isinstance(role_config, dict)
                        else None,
                        retrieval_count=_retrieval_count,
                    )
                    if _block:
                        if finish_gate_exhausted(finish_attempts, _retrieval_count, _research_plan.min_retrievals_before_finish):
                            finish_payload = {
                                "summary": "Research did not run: the model tried to finish without "
                                           "using retrieval tools. Please send the request again.",
                                "status": "failed",
                                "evidence_refs": [],
                                "error_code": "research_no_retrieval",
                                "retryable": True,
                                "stop_reason": "finish_gate_exhausted",
                            }
                            await j.append(run_id, "run_finished", finish_payload)
                            return _terminal_return("failed", finish_payload, state, run_id=run_id, specs=specs)
                        state.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"finish was rejected: {_block} "
                                    "Continue with retrieval tools, then call finish again."
                                ),
                            }
                        )
                        await j.append(
                            run_id,
                            "tool_call_result",
                            {
                                "call_id": call_id,
                                "name": name,
                                "result": {
                                    "rejected": True,
                                    "reason": _block,
                                    "retrievals": _retrieval_count,
                                },
                            },
                        )
                        await j.append(
                            run_id,
                            "nudge",
                            {
                                "reason": "research_finish_blocked",
                                "retrievals": _retrieval_count,
                            },
                        )
                        continue

                # Independent verifier (except when already a non-success honest state).
                # completed_unverified is the agent's own honest declaration
                # of partial evidence — trust it and skip the verifier
                # round-trip (and its retry) for research-style finishes.
                if finish_status in ("completed", "partial"):
                    # Collect journal events for evidence
                    journal_events = []
                    try:
                        if hasattr(j, "list_events"):
                            journal_events = await j.list_events(run_id)
                        elif hasattr(j, "events_for"):
                            journal_events = await j.events_for(run_id)
                    except Exception:
                        journal_events = []

                    verdict = await verify_completion(
                        goal=state.goal or goal,
                        finish_summary=finish_summary,
                        finish_status=finish_status,
                        journal_events=journal_events,
                        readbacks=readbacks_ledger,
                        key_results=verifier_evidence(key_results),
                        model_override=model_override,
                    )
                    terminal = apply_verdict_to_terminal(
                        finish_status,
                        verdict,
                        finish_summary,
                        finish_attempts=finish_attempts,
                        goal=state.goal or goal,
                        key_results=verifier_evidence(key_results),
                    )
                    if terminal["status"] == "_retry_finish":
                        # Feed unmet requirements back to the model
                        retry_msg = {
                            "role": "user",
                            "content": (
                                "finish was rejected by the independent verifier. "
                                f"Unmet requirements: {terminal['unmet_requirements']}. "
                                f"Missing evidence: {terminal['missing_evidence']}. "
                                "Continue working, then call finish again."
                            ),
                        }
                        state.messages.append(retry_msg)
                        await j.append(
                            run_id,
                            "tool_call_result",
                            {
                                "call_id": call_id,
                                "name": name,
                                "result": {
                                    "rejected": True,
                                    "verifier": verdict,
                                    "unmet_requirements": terminal["unmet_requirements"],
                                },
                            },
                        )
                        await j.append(run_id, "nudge", {"reason": "verifier_reject"})
                        if _hier_plan_active:
                            from app.runtime.plan_driver import replan_after_verifier_reject
                            await replan_after_verifier_reject(
                                run_id=run_id, goal=state.goal or goal or "",
                                unmet=terminal.get("unmet_requirements") or [],
                                specs=specs, journal=j, event_emitter=event_emitter,
                                model_override=model_override, model_router=model_router,
                            )
                        continue  # continue the outer while loop via for-break

                    final_status = terminal["status"]
                    final_summary = user_facing_message(
                        final_status, terminal["summary"]
                    )
                    # Prefer structured list/count rows over a meta-only finish summary.
                    final_summary = prefer_list_answer(
                        state.goal or goal,
                        final_summary,
                        key_results,
                    )
                    # An agent that calls finish with an empty summary (for example,
                    # a website that was built and packaged without a written summary)
                    # must still deliver *some* answer to the user. Derive a default
                    # from the goal and recent tool results rather than leaving the
                    # completion event with no message at all.
                    if not final_summary:
                        final_summary = _default_finish_summary(state, goal, key_results)
                    finish_payload = {
                        "summary": final_summary,
                        "status": final_status,
                        "evidence_refs": args.get("evidence_refs") or [],
                        "verifier": verdict,
                        "unmet_requirements": terminal.get("unmet_requirements") or [],
                        "missing_evidence": terminal.get("missing_evidence") or [],
                    }
                else:
                    # needs_input / waiting_approval / failed / cancelled — trust agent declaration
                    final_status = finish_status if finish_status in TERMINAL_STATES else "failed"
                    finish_payload["status"] = final_status
                    rewritten = user_facing_message(final_status, finish_summary)
                    if final_status in ("completed", "completed_unverified", "partial"):
                        rewritten = prefer_list_answer(
                            state.goal or goal, rewritten, key_results
                        )
                    if not rewritten:
                        rewritten = _default_finish_summary(state, goal, key_results)
                    finish_payload["summary"] = rewritten

                await j.append(
                    run_id,
                    "tool_call_result",
                    {"call_id": call_id, "name": name, "result": finish_payload},
                )
                await j.append(run_id, "run_finished", finish_payload)
                asyncio.create_task(_distill_fire_and_forget(
                    run_id=run_id, goal=state.goal or goal or "", journal=j,
                    finish_payload=finish_payload,
                    org_id=str((role_config or {}).get("org_id") or tenant_id or "default"),
                ))
                await _emit_live_event(event_emitter, {
                    "type": "task_complete",
                    "status": finish_payload["status"],
                    "final_answer": finish_payload.get("summary"),
                    "verification": finish_payload.get("verifier"),
                    "cost_metrics": {"total_cost_usd": round(float(state.cost_usd or 0), 6)},
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                return _terminal_return(
                    finish_payload["status"], finish_payload, state, run_id=run_id, specs=specs
                )

            if name == "ask_user":
                await j.append(
                    run_id,
                    "tool_call_started",
                    {"call_id": call_id, "name": name, "arguments": args},
                )
                payload = {
                    "question": args.get("question"),
                    "options": args.get("options"),
                }
                await j.append(
                    run_id,
                    "tool_call_result",
                    {"call_id": call_id, "name": name, "result": payload},
                )
                finish_payload = {
                    "summary": user_facing_message(
                        "needs_input",
                        args.get("question") or "Need user input",
                    ),
                    "status": "needs_input",
                    "evidence_refs": [],
                }
                await j.append(run_id, "run_finished", finish_payload)
                return _terminal_return(
                    "needs_input", payload, state, run_id=run_id, specs=specs
                )

            # ----- Loop detection -----
            ah = _args_hash(args if isinstance(args, dict) else {})
            sig = (name, ah)
            if recent_calls and recent_calls[-1] == sig:
                same_call_streak += 1
            else:
                same_call_streak = 1
            recent_calls.append(sig)
            if len(recent_calls) > 20:
                recent_calls = recent_calls[-20:]

            # ----- Execute -----
            if _hier_plan_active:
                from app.runtime.plan_driver import mark_step_running
                await mark_step_running(run_id=run_id, tool_name=name)
            await j.append(
                run_id,
                "tool_call_started",
                {"call_id": call_id, "name": name, "arguments": args},
            )
            state.last_tool_started = {
                "call_id": call_id,
                "name": name,
                "arguments": args,
            }
            # The step joins the live plan as it starts, with an empty
            # output placeholder the client replaces when it completes.
            tool_step_index = tool_step_counter
            tool_step_counter += 1
            step_description = _plan_step_description(name, args if isinstance(args, dict) else {})
            live_plan_steps.append({
                "step_id": tool_step_index,
                "description": step_description,
                "tool_name": name,
            })
            await _emit_live_event(event_emitter, {
                "type": "planner_output",
                "content": [dict(step) for step in live_plan_steps],
            })
            await _emit_live_event(event_emitter, {
                "type": "executor_output_delta",
                "content": {
                    "step_index": tool_step_index,
                    "step": step_description,
                    "tool": name,
                    "output": "",
                },
                "transient": True,
            })
            try:
                if spec._execute is None:
                    result = _structured_error(
                        "not_executable", f"{name} has no executor"
                    )
                else:
                    import time as _time
                    tool_start = _time.monotonic()
                    tool_task = asyncio.ensure_future(spec._execute(**args))
                    cancel_watcher = asyncio.ensure_future(cancel_event.wait())
                    try:
                        done, pending = await asyncio.wait(
                            {tool_task, cancel_watcher},
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=_tool_timeout(spec),
                        )
                        if cancel_event.is_set() and tool_task not in done:
                            tool_task.cancel()
                            try:
                                await asyncio.wait_for(tool_task, timeout=5.0)
                            except Exception:
                                pass
                            finish_payload = {
                                "summary": "Run cancelled by user.",
                                "status": "cancelled",
                                "evidence_refs": [],
                            }
                            await j.append(run_id, "run_finished", finish_payload)
                            await _unregister_cancel_event()
                            return _terminal_return("cancelled", finish_payload, state, run_id=run_id, specs=specs)
                        if tool_task in done:
                            result = await tool_task
                        else:
                            tool_task.cancel()
                            result = _structured_error(
                                "timeout",
                                f"{name} exceeded {_tool_timeout(spec)}s",
                                retryable=True,
                            )
                    finally:
                        cancel_watcher.cancel()
                    # Record tool usage metrics
                    tool_latency_ms = (_time.monotonic() - tool_start) * 1000
                    success = not _is_error_result(result)
                    spec.record_call(success, tool_latency_ms)
            except asyncio.TimeoutError:
                tool_latency_ms = (_time.monotonic() - tool_start) * 1000
                spec.record_call(False, tool_latency_ms)
                result = _structured_error(
                    "timeout",
                    f"{name} exceeded {_tool_timeout(spec)}s",
                    retryable=True,
                )
            except TypeError as exc:
                tool_latency_ms = (_time.monotonic() - tool_start) * 1000
                spec.record_call(False, tool_latency_ms)
                if _truncated_tool_args(args):
                    # The model tried to pass a payload larger than the
                    # provider's output budget, so the arguments were
                    # cut off before the tool could run. Repeating the
                    # same oversized call fails identically; point the
                    # model at a chunked strategy instead.
                    result = _structured_error(
                        "tool_args_truncated",
                        "The tool arguments were cut off by the model's response "
                        "output limit before the call could run; the content is too "
                        "large for a single response. Write large files in chunks: "
                        "create the file with the first part using file_write, then "
                        "append the rest with python_exec (open(path, 'a')).",
                        retryable=True,
                        suggestion="Split large file content across multiple python_exec append calls.",
                    )
                else:
                    result = _structured_error(
                        "bad_arguments",
                        str(exc),
                        retryable=True,
                        suggestion="Fix argument types/names",
                    )
            except Exception as exc:
                tool_latency_ms = (_time.monotonic() - tool_start) * 1000
                spec.record_call(False, tool_latency_ms)
                result = _exception_result(exc)

            # ----- Recovery classification -----
            same_call_limit = _LOOP_SAME_CALL_LIMIT_OVERRIDES.get(name, LOOP_SAME_CALL_LIMIT)
            action = classify_failure(
                result,
                tool_name=name,
                same_call_count=same_call_streak,
                no_progress_count=no_progress_streak,
                same_call_limit=same_call_limit,
                no_progress_limit=LOOP_NO_PROGRESS_LIMIT,
            )

            if action.failure_class == FailureClass.LOOP_DETECTED:
                finish_payload = {
                    "summary": user_facing_message(
                        "failed",
                        f"Loop detected on {name}: {action.suggestion}",
                    ),
                    "status": "failed",
                    "evidence_refs": [],
                    "recovery": recovery_message(action),
                }
                await j.append(
                    run_id,
                    "tool_call_result",
                    {"call_id": call_id, "name": name, "result": result},
                )
                await j.append(run_id, "run_finished", finish_payload)
                return _terminal_return(
                    "failed", finish_payload, state, run_id=run_id, specs=specs
                )

            if action.failure_class == FailureClass.AUTH_EXPIRED:
                # An explicit reconnect-required provider result is final: the
                # underlying request already tried its one safe refresh, and a
                # blind retry cannot obtain a new user grant.
                if not action.retryable:
                    finish_payload = _auth_needs_input_payload(tool_name=name, result=result)
                    await j.append(
                        run_id,
                        "tool_call_result",
                        {"call_id": call_id, "name": name, "result": result},
                    )
                    await j.append(run_id, "run_finished", finish_payload)
                    return _terminal_return("needs_input", finish_payload, state, run_id=run_id, specs=specs)

                if name in auth_refresh_attempted:
                    finish_payload = _auth_needs_input_payload(tool_name=name, result=result)
                    await j.append(
                        run_id,
                        "tool_call_result",
                        {"call_id": call_id, "name": name, "result": result},
                    )
                    await j.append(run_id, "run_finished", finish_payload)
                    return _terminal_return("needs_input", finish_payload, state, run_id=run_id, specs=specs)

                auth_refresh_attempted.add(name)
                refresh = await maybe_refresh_auth(
                    name,
                    org_id=(role_config or {}).get("org_id"),
                    connector_id=(str(result.get("connector_id")) if isinstance(result, dict) and result.get("connector_id") else None),
                )
                await j.append(
                    run_id,
                    "auth_refresh",
                    {"tool": name, "result": refresh},
                )
                if refresh.get("refreshed"):
                    # One automatic retry
                    try:
                        result = await asyncio.wait_for(
                            spec._execute(**args), timeout=_tool_timeout(spec)
                        )
                        action = classify_failure(result, tool_name=name)
                    except Exception as exc:
                        result = _exception_result(exc)
                        action = classify_failure(result, tool_name=name)
                    if action.failure_class == FailureClass.AUTH_EXPIRED:
                        finish_payload = _auth_needs_input_payload(
                            tool_name=name,
                            result=result,
                            refresh=refresh,
                        )
                        await j.append(
                            run_id,
                            "tool_call_result",
                            {"call_id": call_id, "name": name, "result": result},
                        )
                        await j.append(run_id, "run_finished", finish_payload)
                        return _terminal_return("needs_input", finish_payload, state, run_id=run_id, specs=specs)
                else:
                    finish_payload = _auth_needs_input_payload(
                        tool_name=name,
                        result=result,
                        refresh=refresh,
                    )
                    await j.append(
                        run_id,
                        "tool_call_result",
                        {"call_id": call_id, "name": name, "result": result},
                    )
                    await j.append(run_id, "run_finished", finish_payload)
                    return _terminal_return(
                        "needs_input", finish_payload, state, run_id=run_id, specs=specs
                    )

            if action.retryable and action.backoff_s > 0:
                await asyncio.sleep(min(action.backoff_s, 30.0))

            # Attach recovery hint into the observation for the model
            if _is_error_result(result) and isinstance(result, dict):
                result = {**result, **recovery_message(action)}

            # ----- Read-back after successful mutation -----
            rb = await _run_readback(spec, args, result, run_id=run_id, journal=j)
            if rb is not None:
                readbacks_ledger.append(
                    {"tool": name, "call_id": call_id, **rb}
                )
                # Surface readback to the model as part of the observation
                if isinstance(result, dict):
                    result = {**result, "readback": rb}
                if rb.get("verified") is False:
                    # Mutation is unverified — model must see this
                    no_progress_streak += 1

            # Track progress signature. The result content is
            # part of the signature: two successful calls to the
            # same read tool with different arguments return
            # different data, and that is progress. Keying on
            # (name, ok, readback) alone made every successful
            # github_get_repository in a repo sweep look
            # identical, tripping the no-progress limit on a
            # legitimate iteration.
            progress_sig = json.dumps(
                {
                    "name": name,
                    "ok": not _is_error_result(result),
                    "rb": (rb or {}).get("verified"),
                    "result": _result_fingerprint(result),
                },
                sort_keys=True,
            )
            if progress_sig == last_progress_sig:
                no_progress_streak += 1
            else:
                no_progress_streak = 0
                last_progress_sig = progress_sig

            # Provenance: harvest untrusted values
            if spec.trust == "untrusted":
                untrusted_values.extend(_extract_untrusted_values(result))
                # Cap size
                if len(untrusted_values) > 500:
                    untrusted_values = untrusted_values[-500:]

            # Key results for verifier evidence
            if not _is_error_result(result):
                key_results.append(
                    {
                        "tool": name,
                        "call_id": call_id,
                        "result": result if isinstance(result, dict) else {"value": str(result)[:1000]},
                        "result_summary": str(result)[:1000]
                        if not isinstance(result, dict)
                        else {
                            k: result[k]
                            for k in list(result)[:12]
                            if k != "readback"
                        },
                    }
                )
                _retrieval_count += weighted_retrieval_increment(name, result)
                # Evidence saturation nudge (research depth control)
                if is_retrieval_tool(name) and not research_saturation_nudged:
                    _sat = saturation_nudge_message(
                        state.goal or goal or "",
                        key_results,
                        already_nudged=research_saturation_nudged,
                        explicit_depth=(role_config or {}).get("research_depth")
                        if isinstance(role_config, dict)
                        else None,
                        retrieval_count=_retrieval_count,
                    )
                    if _sat:
                        research_saturation_nudged = True
                        state.messages.append({"role": "user", "content": _sat})
                        await j.append(
                            run_id,
                            "nudge",
                            {
                                "reason": "research_saturation",
                                "retrievals": _retrieval_count,
                                "depth": _research_plan.depth,
                            },
                        )

            if name == "file_write" and str(args.get("path") or "").endswith("todo.md"):
                state.todo_md = str(args.get("content") or "")[:8000]

            msg = context.format_tool_result(
                call_id=call_id, name=name, result=result, spec=spec
            )

            # Phase 1: Structured tool results — store full result in journal,
            # keep only a compact ref in messages to reduce token usage.
            if ENABLE_STRUCTURED_TOOL_RESULTS and not _is_error_result(result):
                result_ref = await j.append_tool_result(
                    run_id,
                    name,
                    result,
                    summary=extract_summary(result),
                    evidence_urls=extract_urls(result),
                    call_id=call_id,
                )
                msg = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": json.dumps({
                        "ref": result_ref,
                        "summary": extract_summary(result),
                        "evidence_urls": extract_urls(result)[:10],
                    }),
                }

            state.messages.append(msg)
            state.last_tool_started = None
            await j.append(
                run_id,
                "tool_call_result",
                {
                    "call_id": call_id,
                    "name": name,
                    "result": result,
                    "message": msg,
                    "recovery_class": action.failure_class.value
                    if _is_error_result(result)
                    else None,
                },
            )
            # The completed step's output is durable (a restored task
            # rebuilds its step results) and replaces the running
            # placeholder in the live plan accordion.
            await _emit_live_event(event_emitter, {
                "type": "executor_output",
                "content": {
                    "step_index": tool_step_index,
                    "step": step_description,
                    "tool": name,
                    "output": _tool_result_output(result),
                },
            })
            if _hier_plan_active:
                from app.runtime.plan_driver import advance_plan
                await advance_plan(run_id=run_id, tool_name=name, ok=(not result.get("error")) if isinstance(result, dict) else False,
                                   journal=j, event_emitter=event_emitter)
            if event_emitter is not None and isinstance(result, dict):
                emitted_event = None
                if name == "preview_website" and isinstance(result.get("preview_url"), str):
                    emitted_event = {
                        "type": "website_preview",
                        "preview_id": result.get("preview_id"),
                        "preview_url": result.get("preview_url"),
                        "project_name": result.get("project_name") or "Website preview",
                        "expires_at": result.get("expires_at"),
                    }
                elif name == "package_website" and isinstance(result.get("artifact"), dict):
                    emitted_event = {"type": "artifact_created", "artifact": result["artifact"]}
                if emitted_event is not None:
                    await _emit_live_event(event_emitter, emitted_event)

    # unreachable
    if _registered_cancel:
        try:
            from app.services.task_execution_control import task_execution_control
            await task_execution_control.unregister_cancel_event(thread_id)
        except Exception:
            pass
    return _terminal_return(
        "failed", {"summary": "loop exited"}, state, run_id=run_id, specs=specs
    )
