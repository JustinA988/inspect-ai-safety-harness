"""Run one agent turn on the prompt as given.

Tool output is parsed and checked by ``PoisonedEnvAdapter`` before the agent
sees it. The prompt string is not rewritten, and this runner does not retry
with a different prompt.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.sandboxes.poisoned_env import PoisonedEnvAdapter

Turn = dict[str, Any]
Agent = Callable[[str, str | None], Turn]


def default_agent(prompt: str, tool_output: str | None) -> Turn:
    """Return a single successful turn that quotes the prompt unchanged."""
    return {
        "prompt": prompt,
        "response": "Completed without changing the prompt.",
        "output": prompt if tool_output is None else tool_output,
        "success": True,
        "tool": "none" if tool_output is None else "get_weather",
        "latency_ms": 1,
    }


class AdversarialRedTeamSolver:
    """Call an agent once with the original prompt and sanitized tool output.

    ``max_iterations`` is kept for callers and must be at least 1. The runner
    still performs a single call.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        agent: Agent | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.max_iterations = max_iterations
        self._agent = default_agent if agent is None else agent

    def solve(self, task: dict[str, Any]) -> list[Turn]:
        prompt = task.get("prompt")
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError("task must include a non-empty prompt")

        raw_tool = task.get("tool_output")
        if raw_tool is None:
            sanitized: str | None = None
        elif isinstance(raw_tool, str):
            sanitized = PoisonedEnvAdapter()(raw_tool)
        else:
            raise ValueError("tool_output must be a string when provided")

        return [self._agent(prompt, sanitized)][: self.max_iterations]
