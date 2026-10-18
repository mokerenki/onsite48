"""LangGraph entry — Phase 3 cutover routes agent runs to the runtime loop.

Provides:
  - run_agent / invoke_graph: direct loop entry
  - create_graph: LangGraph-compatible graph for websocket_handler and tests
"""
from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger("onsite48.graph")


async def run_agent(
    run_id: str,
    goal: str,
    role_config: dict | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Single entry point used by workers and API."""
    from app.runtime.loop import run as loop_run

    return await loop_run(run_id, goal, role_config or {}, **kwargs)


async def invoke_graph(*args: Any, **kwargs: Any) -> dict[str, Any]:
    if args:
        run_id = args[0] if len(args) > 0 else kwargs.get("run_id", "unknown")
        goal = args[1] if len(args) > 1 else kwargs.get("goal", "")
        role_config = args[2] if len(args) > 2 else kwargs.get("role_config")
        return await run_agent(
            str(run_id),
            str(goal),
            role_config,
            **{k: v for k, v in kwargs.items() if k not in ("run_id", "goal", "role_config")},
        )
    return await run_agent(
        str(kwargs.get("run_id") or "unknown"),
        str(kwargs.get("goal") or ""),
        kwargs.get("role_config"),
        **{k: v for k, v in kwargs.items() if k not in ("run_id", "goal", "role_config")},
    )


def create_graph(
    model_router: Any,
    domain_router: Any,
    tool_registry: Any,
    checkpointer: Any = None,
    live_event_emitter: Any = None,
):
    """Build a minimal LangGraph that delegates to the runtime loop.

    websocket_handler expects graph.astream / aget_state / aupdate_state.
    We compile a single-node graph whose node invokes app.runtime.loop.run.

    ``live_event_emitter`` receives UI events (plan growth, tool output,
    running cost) while the loop runs, so the websocket stream can
    forward them as they happen instead of holding them until the node
    returns at completion.
    """
    try:
        from langgraph.graph import END, StateGraph
        from typing_extensions import TypedDict
    except ImportError:
        # Fallback: object with astream that yields a single final result
        return _LoopGraphShim(model_router, domain_router, tool_registry, checkpointer, live_event_emitter)

    class AgentState(TypedDict, total=False):
        task_id: str
        thread_id: str
        task: str
        user_id: str
        tenant_id: str
        messages: list
        plan: list
        current_step: int
        results: list
        prior_results: list
        tool_calls: list
        verification: Any
        needs_replan: bool
        done: bool
        final_answer: Any
        status: str
        cost_metrics: dict
        scratchpad: str
        confirmed_actions: dict
        skipped_steps: dict
        conversation_context: str
        task_metadata: dict
        is_follow_up: bool
        research_depth: str
        confirmed_actions: dict
        rejected_tool_call: dict
        waiting_confirmation: dict
        website_events: list
        schedule_authorization: dict

    async def loop_node(state: AgentState) -> dict:
        from app.runtime.loop import run as loop_run

        goal = str(state.get("task") or "")
        run_id = str(state.get("task_id") or state.get("thread_id") or "unknown")
        website_events: list[dict[str, Any]] = []

        async def capture_website_event(event: dict[str, Any]) -> None:
            website_events.append(event)
            if live_event_emitter is None:
                return
            try:
                forwarded = live_event_emitter(event)
                if hasattr(forwarded, "__await__"):
                    await forwarded
            except Exception as exc:
                logger.warning("graph.live_event_forward_failed", error=str(exc)[:200])

        try:
            session_factory = None
            try:
                from app.db import get_session_factory

                session_factory = get_session_factory()
            except Exception:
                session_factory = None
            
            # Pass follow-up context through role_config so loop.run can use it
            loop_role_config = dict(state.get("role_config", {}) or {})
            loop_role_config["conversation_context"] = state.get("conversation_context", "")
            loop_role_config["is_follow_up"] = bool(state.get("is_follow_up", False))
            loop_role_config["research_depth"] = state.get("research_depth")
            return await loop_run(
                run_id,
                goal,
                loop_role_config,
                model_router=model_router,
                tool_registry=tool_registry,
                registry=tool_registry,
                thread_id=state.get("thread_id"),
                user_id=state.get("user_id"),
                tenant_id=state.get("tenant_id"),
                session_factory=session_factory,
                confirmed_actions=state.get("confirmed_actions") or {},
                rejected_tool_call=state.get("rejected_tool_call") or {},
                schedule_authorization=state.get("schedule_authorization"),
                event_emitter=capture_website_event,
            )
        except Exception as exc:
            # Do not re-execute on TypeError — fail once and surface the error
            logger.exception("graph.loop_node_failed", error=str(exc)[:300])
            return {
                "done": True,
                "status": "failed",
                "final_answer": f"Agent loop failed: {exc}",
                "results": [{"error": str(exc)}],
            }

        status = "completed"
        payload: dict[str, Any] = {}
        final_answer = None
        results: list = []
        if isinstance(result, dict):
            status = str(result.get("status") or "completed")
            payload = result.get("result") or result.get("finish_payload") or {}
            if isinstance(payload, dict):
                # Preserve empty strings. An agent that calls finish with an
                # empty summary (for example, a website that was built and
                # packaged without a written summary) must still deliver *some*
                # answer to the user; collapsing the empty string to None via an
                # `or` chain leaves final_answer unset and the completion event
                # carries no message at all.
                final_answer = payload.get("summary")
                if final_answer is None:
                    final_answer = payload.get("final_answer")
                if final_answer is None:
                    final_answer = payload.get("message")
                results = payload.get("results") or [payload]
            else:
                final_answer = str(payload) if payload else str(result.get("message") or "")
                results = [result]
        else:
            final_answer = str(result)
            results = [{"output": final_answer}]

        done = status in (
            "completed",
            "completed_unverified",
            "failed",
            "cancelled",
            "partial",
            "needs_input",
            "waiting_approval",
        )
        return {
            "done": done,
            "status": status,
            "final_answer": final_answer,
            "results": results if isinstance(results, list) else [results],
            "waiting_confirmation": payload.get("pending_call") if isinstance(payload, dict) else None,
            "step_count": result.get("step_count") if isinstance(result, dict) else 0,
            "website_events": website_events,
            "confirmed_actions": {},
            "rejected_tool_call": {},
        }

    builder = StateGraph(AgentState)
    builder.add_node("loop", loop_node)
    # Alias nodes so aupdate_state(as_node="planner") from websocket still works
    builder.add_node("planner", loop_node)
    builder.set_entry_point("loop")
    builder.add_edge("loop", END)
    builder.add_edge("planner", END)

    compiled = builder.compile(checkpointer=checkpointer)
    logger.info("graph.create_graph_compiled", entry="loop")
    return compiled


class _LoopGraphShim:
    """Minimal stand-in when langgraph is unavailable."""

    def __init__(self, model_router, domain_router, tool_registry, checkpointer, live_event_emitter=None):
        self.model_router = model_router
        self.tool_registry = tool_registry
        self.checkpointer = checkpointer
        self.live_event_emitter = live_event_emitter
        self._state: dict[str, Any] = {}

    async def astream(self, initial_state, config=None):
        if initial_state is not None and isinstance(initial_state, dict):
            self._state.update(initial_state)
        from app.runtime.loop import run as loop_run

        goal = str(self._state.get("task") or "")
        run_id = str(self._state.get("task_id") or self._state.get("thread_id") or "unknown")
        website_events: list[dict[str, Any]] = []
        try:
            async def capture_website_event(event: dict[str, Any]) -> None:
                website_events.append(event)
                if self.live_event_emitter is None:
                    return
                try:
                    forwarded = self.live_event_emitter(event)
                    if hasattr(forwarded, "__await__"):
                        await forwarded
                except Exception:
                    pass

            # Pass follow-up context through role_config so loop.run can use it
            shim_role_config = dict(self._state.get("role_config", {}) or {})
            shim_role_config["conversation_context"] = self._state.get("conversation_context", "")
            shim_role_config["is_follow_up"] = bool(self._state.get("is_follow_up", False))
            shim_role_config["research_depth"] = self._state.get("research_depth")

            result = await loop_run(
                run_id,
                goal,
                shim_role_config,
                thread_id=self._state.get("thread_id"),
                user_id=self._state.get("user_id"),
                tenant_id=self._state.get("tenant_id"),
                confirmed_actions=self._state.get("confirmed_actions") or {},
                rejected_tool_call=self._state.get("rejected_tool_call") or {},
                schedule_authorization=self._state.get("schedule_authorization"),
                event_emitter=capture_website_event,
            )
        except Exception as exc:
            result = {"status": "failed", "result": {"summary": str(exc)}}
        status = result.get("status", "completed") if isinstance(result, dict) else "completed"
        final = None
        if isinstance(result, dict):
            payload = result.get("result") or {}
            final = payload.get("summary") if isinstance(payload, dict) else str(payload)
        self._state["done"] = True
        self._state["status"] = status
        self._state["final_answer"] = final
        self._state["results"] = [result]
        self._state["step_count"] = result.get("step_count", 0) if isinstance(result, dict) else 0
        self._state["waiting_confirmation"] = ((result.get("result") or {}).get("pending_call") if isinstance(result, dict) and isinstance(result.get("result"), dict) else None)
        self._state["confirmed_actions"] = {}
        self._state["rejected_tool_call"] = {}
        self._state["website_events"] = website_events
        yield {"loop": dict(self._state)}

    async def aget_state(self, config=None):
        class _Snap:
            def __init__(self, values):
                self.values = values
        return _Snap(dict(self._state))

    async def aupdate_state(self, config, values, as_node=None):
        if isinstance(values, dict):
            self._state.update(values)
