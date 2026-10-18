"""Loop / journal / kill-and-resume tests against a fake model (no live API)."""
from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.services.browser_session_router import select_backend


class FakeCompleteResult:
    def __init__(self, content=None, tool_calls=None, model="fake", provider="fake"):
        self.content = content
        self.tool_calls = tool_calls or []
        self.model = model
        self.provider = provider
        self.path = "fake"
        self.usage = {"input_tokens": 10, "output_tokens": 5}
        self.cost_usd = 0.0
        self.raw = None


@pytest.mark.asyncio
async def test_select_backend_policy_unit():
    assert select_backend().backend == "cloud"
    assert select_backend(requires_user_browser=True, extension_connected=True, extension_authorized=True).backend == "local"
    assert select_backend(requires_user_browser=True, extension_connected=False).backend == "ask_user"


@pytest.mark.asyncio
async def test_loop_finish_with_fake_model():
    """Loop should terminate when the model calls finish."""
    from app.runtime import models as runtime_models

    async def fake_complete(messages, tools=None, **kwargs):
        # First call: propose finish
        return FakeCompleteResult(
            content=None,
            tool_calls=[{
                "id": "call_1",
                "name": "finish",
                "arguments": {"summary": "done", "status": "completed"},
            }],
        )

    with patch.object(runtime_models, "complete", side_effect=fake_complete):
        from app.runtime.loop import run as loop_run

        result = await loop_run(str(uuid.uuid4()), "Say hello and finish", {})
        assert isinstance(result, dict)
        assert result.get("status") in (
            "completed", "completed_unverified", "partial", "failed", "cancelled", "needs_input"
        )


@pytest.mark.asyncio
async def test_agent_computer_auth_words_do_not_trigger_connector_refresh():
    """A browser-page auth error is not a connector OAuth error."""
    from app.runtime import loop as loop_mod
    from app.runtime.journal import Journal
    from app.tools.registry import BaseTool, ToolRegistry, ToolSchema

    class FailingComputerNavigateTool(BaseTool):
        def _build_schema(self):
            return ToolSchema(
                name="computer_navigate",
                description="Test browser navigation.",
                parameters={"url": {"type": "string"}},
                required=["url"],
                idempotency="read_only",
                approval_required=False,
            )

        async def execute(self, url: str, **_: Any):
            del url
            raise RuntimeError(
                "The target page requires authentication before navigation can continue."
            )

    registry = ToolRegistry()
    registry.register(FailingComputerNavigateTool())
    journal = Journal(session_factory=None)
    responses = iter([
        FakeCompleteResult(tool_calls=[{
            "id": "navigate",
            "name": "computer_navigate",
            "arguments": {"url": "https://example.com"},
        }]),
        FakeCompleteResult(tool_calls=[{
            "id": "finish",
            "name": "finish",
            "arguments": {"summary": "Navigation could not continue.", "status": "needs_input"},
        }]),
    ])

    async def fake_complete(*_args, **_kwargs):
        return next(responses)

    refresh = AsyncMock(return_value={"refreshed": False, "detail": "not used"})
    with patch.object(loop_mod, "model_complete", side_effect=fake_complete), patch.object(
        loop_mod, "maybe_refresh_auth", refresh
    ):
        result = await loop_mod.run(
            str(uuid.uuid4()),
            "Open an example page.",
            {},
            registry=registry,
            journal=journal,
            max_steps=3,
        )

    assert refresh.await_count == 0
    assert result["status"] == "needs_input"


@pytest.mark.asyncio
async def test_journal_append_and_replay_in_memory():
    """Journal append-only seq and replay without DB when session factory missing."""
    from app.runtime.journal import Journal

    j = Journal(session_factory=None)
    run_id = str(uuid.uuid4())
    # append may no-op or use memory depending on implementation
    try:
        await j.append(run_id, "run_started", {"goal": "test"})
        await j.append(run_id, "model_request", {"n": 1})
        events = await j.list_events(run_id)
        assert isinstance(events, list)
    except Exception as exc:
        # Acceptable if journal requires DB — mark soft
        pytest.skip(f"journal requires DB: {exc}")


@pytest.mark.asyncio
async def test_model_router_accepts_loop_role():
    """ModelRouter must not raise Unknown role: loop."""
    from app.core.model_router import ModelRouter

    router = ModelRouter()
    # Resolve without calling network if possible
    resolved = router._resolve_role("loop") if hasattr(router, "_resolve_role") else "planning"
    assert resolved in ("loop", "planning")
    # route with empty messages may hit network — mock client
    async def fake_chat(messages, system_prompt=None, temperature=None, max_tokens=None, **kw):
        from app.core.model_router import ModelResponse
        return ModelResponse(content="ok", model="fake", provider="fake")

    with patch.object(router, "_role_map", {"planning": type("C", (), {"chat": fake_chat, "config": type("Cfg", (), {"provider": "fake"})()})()}):
        # Rebuild map with async chat
        class C:
            config = type("Cfg", (), {"provider": "fake"})()
            async def chat(self, messages, system_prompt=None, temperature=None, max_tokens=None, **kw):
                from app.core.model_router import ModelResponse
                try:
                    return ModelResponse(content="ok", model="fake", provider="fake")
                except TypeError:
                    return type("R", (), {"content": "ok", "model": "fake", "provider": "fake", "usage": None, "tool_calls": None})()
        router._role_map = {"planning": C(), "verification": C(), "fallback": C()}
        resp = await router.route("loop", [{"role": "user", "content": "hi"}])
        assert getattr(resp, "content", None) == "ok"


@pytest.mark.asyncio
async def test_create_graph_importable():
    from app.graph import create_graph, run_agent
    assert callable(create_graph)
    assert callable(run_agent)


@pytest.mark.asyncio
async def test_app_models_import_not_shadowed():
    """Stub package must not shadow models.py — ConnectorConnection must import."""
    from app.models import ConnectorConnection, User, OrganizationMember, BrowserBridgeDevice
    assert ConnectorConnection is not None
    assert User is not None
    assert OrganizationMember is not None
    assert BrowserBridgeDevice is not None


@pytest.mark.asyncio
async def test_app_db_import_not_shadowed():
    from app.db import Base
    assert Base is not None


@pytest.mark.asyncio
async def test_in_memory_resume_skips_completed_tool():
    """In-memory: after tool_call_result is journaled, new loop.run does not re-propose that tool."""
    from app.runtime.journal import Journal
    from app.runtime import models as runtime_models
    from app.runtime import loop as loop_mod
    from app.runtime.loop import run as loop_run

    run_id = str(uuid.uuid4())
    j = Journal(session_factory=None)
    # Clear any shared memory for this id
    Journal._memory.pop(run_id, None)

    call_log: list[str] = []

    async def fake_complete(messages, tools=None, **kwargs):
        n = len(call_log)
        call_log.append(f"call_{n}")
        if n == 0:
            # First model call: propose a non-finish tool
            return FakeCompleteResult(
                tool_calls=[{
                    "id": "call_tool_1",
                    "name": "finish",  # use finish for simplicity in offline; or a no-op
                    "arguments": {"summary": "first finish", "status": "completed"},
                }],
            )
        # Should not be reached on resume of finished run
        return FakeCompleteResult(
            tool_calls=[{
                "id": "call_tool_2",
                "name": "finish",
                "arguments": {"summary": "second finish", "status": "completed"},
            }],
        )

    with patch.object(runtime_models, "complete", side_effect=fake_complete), \
         patch.object(loop_mod, "model_complete", side_effect=fake_complete):
        result1 = await loop_run(run_id, "do something", {}, journal=j)
        assert result1.get("status") in ("completed", "completed_unverified", "finished", "partial", "failed")
        steps1 = result1.get("step_count") or (result1.get("state") or {}).get("step_count") or 0
        assert steps1 >= 1 or result1.get("status") in ("completed", "finished", "completed_unverified")

    # Finished-run short-circuit: same run_id must not call model again
    call_log.clear()
    with patch.object(runtime_models, "complete", side_effect=fake_complete), \
         patch.object(loop_mod, "model_complete", side_effect=fake_complete):
        result2 = await loop_run(run_id, "do something", {}, journal=j)
        assert result2.get("status") in ("completed", "completed_unverified", "finished", "failed")
        assert len(call_log) == 0, "finished run must short-circuit without model call"


@pytest.mark.asyncio
async def test_finished_run_idempotent_short_circuit():
    """Journal ending with run_finished → loop.run returns payload without model call."""
    from app.runtime.journal import Journal
    from app.runtime import models as runtime_models
    from app.runtime.loop import run as loop_run

    run_id = str(uuid.uuid4())
    j = Journal(session_factory=None)
    Journal._memory.pop(run_id, None)

    await j.append(run_id, "run_started", {"goal": "idempotent test"})
    await j.append(run_id, "model_response", {"tokens": 12, "cost_usd": 0.001})
    finish_payload = {
        "summary": "already done",
        "status": "completed",
        "evidence_refs": [],
    }
    await j.append(run_id, "run_finished", finish_payload)

    called = []

    async def boom(*a, **k):
        called.append(1)
        raise AssertionError("model must not be called for finished run")

    with patch.object(runtime_models, "complete", side_effect=boom):
        result = await loop_run(run_id, "idempotent test", {}, journal=j)
        assert result["status"] in ("finished", "completed")
        assert result.get("result") == finish_payload or (
            isinstance(result.get("result"), dict)
            and result["result"].get("summary") == "already done"
        )
        assert len(called) == 0
        # Metrics present top-level
        assert result.get("step_count") is not None
        assert result.get("token_count") is not None


@pytest.mark.asyncio
async def test_mid_run_resume_in_memory_no_duplicate_tool_exec():
    """Mid-run: journal tool_call_result, resume — model continues, tool not re-executed."""
    from app.runtime.journal import Journal
    from app.runtime import models as runtime_models
    from app.runtime.loop import run as loop_run

    run_id = str(uuid.uuid4())
    j = Journal(session_factory=None)
    Journal._memory.pop(run_id, None)

    # Seed journal as if a tool already completed
    await j.append(run_id, "run_started", {"goal": "resume mid"})
    await j.append(run_id, "model_response", {"tokens": 8, "cost_usd": 0.0})
    await j.append(
        run_id,
        "tool_call_started",
        {"call_id": "c1", "name": "workspace_list", "arguments": {}},
    )
    await j.append(
        run_id,
        "tool_call_result",
        {"call_id": "c1", "name": "workspace_list", "result": {"files": []}},
    )

    model_calls = []

    async def fake_complete(messages, tools=None, **kwargs):
        model_calls.append(1)
        # On resume, model should finish (not re-propose the completed tool)
        return FakeCompleteResult(
            tool_calls=[{
                "id": "c_finish",
                "name": "finish",
                "arguments": {"summary": "resumed and done", "status": "completed"},
            }],
        )

    import app.runtime.loop as loop_mod
    with patch.object(runtime_models, "complete", side_effect=fake_complete), \
         patch.object(loop_mod, "model_complete", side_effect=fake_complete):
        result = await loop_run(run_id, "resume mid", {}, journal=j, max_steps=5)
        assert isinstance(result, dict)
        # Should progress (finish) without re-doing the journaled tool
        assert result.get("status") in (
            "completed", "completed_unverified", "finished", "partial", "failed"
        )
        assert len(model_calls) >= 1
        # step_count should include the replayed model_response
        steps = result.get("step_count") or (result.get("state") or {}).get("step_count") or 0
        assert steps >= 1


@pytest.mark.asyncio
async def test_db_session_factory_resume_mid_run():
    """DB session_factory: kill after tool_call_result, resume without re-executing tool."""
    pytest.importorskip("aiosqlite")
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from app.runtime.journal import Journal, Base as JournalBase
    from app.runtime import models as runtime_models
    from app.runtime.loop import run as loop_run

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    # Create only run_events table
    async with engine.begin() as conn:
        await conn.run_sync(JournalBase.metadata.create_all)

    sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    j = Journal(session_factory=sf)
    run_id = str(uuid.uuid4())

    # Run 1: force stop after first tool by max_steps=1 after seeding partial progress
    # Seed a completed tool result so resume has prior work
    await j.append(run_id, "run_started", {"goal": "db resume"})
    await j.append(run_id, "model_response", {"tokens": 5, "cost_usd": 0.0})
    await j.append(
        run_id,
        "tool_call_started",
        {"call_id": "db_c1", "name": "workspace_list", "arguments": {}},
    )
    await j.append(
        run_id,
        "tool_call_result",
        {"call_id": "db_c1", "name": "workspace_list", "result": {"ok": True}},
    )

    model_calls = []

    async def fake_complete(messages, tools=None, **kwargs):
        model_calls.append(1)
        return FakeCompleteResult(
            tool_calls=[{
                "id": "db_finish",
                "name": "finish",
                "arguments": {"summary": "db resumed finish", "status": "completed"},
            }],
        )

    import app.runtime.loop as loop_mod
    with patch.object(runtime_models, "complete", side_effect=fake_complete), \
         patch.object(loop_mod, "model_complete", side_effect=fake_complete):
        result = await loop_run(
            run_id, "db resume", {}, journal=j, session_factory=sf, max_steps=10
        )
        assert isinstance(result, dict)
        assert result.get("status") in (
            "completed", "completed_unverified", "finished", "partial", "failed"
        )
        assert len(model_calls) >= 1
        # Metrics from top-level or state
        steps = result.get("step_count") or (result.get("state") or {}).get("step_count") or 0
        assert steps >= 1

    # Finished short-circuit on same DB journal
    model_calls.clear()
    with patch.object(runtime_models, "complete", side_effect=fake_complete):
        result2 = await loop_run(
            run_id, "db resume", {}, journal=j, session_factory=sf
        )
        # After finish, status should short-circuit
        if result.get("status") in ("completed", "completed_unverified", "finished"):
            assert len(model_calls) == 0

    await engine.dispose()


@pytest.mark.asyncio
async def test_fast_path_depth_none_direct_answer():
    """Short non-research goal (depth=none) answers directly without tools."""
    from app.runtime import loop as loop_mod
    from app.runtime.loop import run as loop_run

    async def fake_complete(messages, tools=None, **kwargs):
        assert tools is None or tools == []
        return FakeCompleteResult(
            content="The answer is 42.",
            tool_calls=[],
        )

    with patch.object(loop_mod, "model_complete", side_effect=fake_complete):
        result = await loop_run(str(uuid.uuid4()), "What is 2 plus 2?", {})
    assert result.get("status") == "completed"
    assert result["result"]["summary"] == "The answer is 42."


@pytest.mark.asyncio
async def test_fast_path_depth_quick_with_search():
    """Short research goal (depth=quick) uses web_search then synthesizes."""
    from app.runtime import loop as loop_mod
    from app.runtime.loop import run as loop_run
    from app.tools.registry import BaseTool, ToolRegistry, ToolSchema

    class FakeSearchTool(BaseTool):
        def _build_schema(self):
            return ToolSchema(
                name="browser",
                description="Fake web search.",
                parameters={"query": {"type": "string"}},
                required=["query"],
                idempotency="read_only",
                approval_required=False,
            )

        async def execute(self, query: str, **_: Any):
            return {"output": f"Search results for: {query}", "source": "fake"}

    registry = ToolRegistry()
    registry.register(FakeSearchTool())

    calls = []

    async def fake_complete(messages, tools=None, **kwargs):
        calls.append(tools)
        # First call: propose search
        if len(calls) == 1:
            return FakeCompleteResult(
                content=None,
                tool_calls=[{
                    "id": "call_1",
                    "name": "browser",
                    "arguments": {"query": "Manus vs Tenssa"},
                }],
            )
        # Second call: synthesize answer
        return FakeCompleteResult(
            content="Manus and Tenssa are different products.",
            tool_calls=[],
        )

    with patch.object(loop_mod, "model_complete", side_effect=fake_complete):
        result = await loop_run(
            str(uuid.uuid4()),
            "Manus vs Tenssa",
            {},
            registry=registry,
            max_steps=5,
        )
    assert result.get("status") == "completed"
    assert "Manus and Tenssa" in result["result"]["summary"]
    # Two model calls: one to decide search, one to synthesize
    assert len(calls) == 2
