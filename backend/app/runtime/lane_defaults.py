"""Per-lane model defaults for the agent runtime.

Only models the product actually has clients for (Kimi, DeepSeek).
Do NOT set production defaults from simulated lane-sweep scores.
"""
from __future__ import annotations

import os
from typing import Dict

# Product-supported providers only
KIMI_DEFAULT = os.environ.get("KIMI_MODEL") or "kimi-latest"
DEEPSEEK_DEFAULT = os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"

# Lane → model. loop/worker use primary (Kimi); planner/verifier/summarizer use DeepSeek.
LANE_DEFAULTS: Dict[str, str] = {
    "loop": KIMI_DEFAULT,
    "worker": KIMI_DEFAULT,
    "planner": DEEPSEEK_DEFAULT,
    "verifier": DEEPSEEK_DEFAULT,
    "summarizer": DEEPSEEK_DEFAULT,
    "vision": KIMI_DEFAULT,
}

# Models the router can actually call (no gpt-4o / claude / gemini until clients exist)
SUPPORTED_MODELS = (
    KIMI_DEFAULT,
    DEEPSEEK_DEFAULT,
    "kimi-latest",
    "deepseek-chat",
    "deepseek-reasoner",
)


def model_for_lane(lane: str) -> str:
    return LANE_DEFAULTS.get(lane, KIMI_DEFAULT)


def is_supported_model(name: str) -> bool:
    return name in SUPPORTED_MODELS or name.startswith("kimi") or name.startswith("deepseek")
