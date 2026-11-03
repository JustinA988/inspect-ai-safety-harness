"""Combine correctness, token efficiency, latency, and repeated tool calls."""

from __future__ import annotations

from typing import Any


def _char_len(value: object) -> int:
    return len(str(value))


def _latency_ms(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


class TelemetryScorer:
    """Score a transcript on a 0–100 scale.

    Weights: correctness 50, efficiency 30, speed 20. Repeated tool names
    subtract up to 20. An empty transcript scores 0.
    """

    @staticmethod
    def score(transcript: list[dict[str, Any]]) -> float:
        count = len(transcript)
        if count == 0:
            return 0.0

        correct = sum(1 for turn in transcript if turn.get("success") is True)
        correctness = correct / count

        total_tokens = sum(_char_len(turn.get("response", "")) for turn in transcript)
        output_tokens = sum(
            _char_len(turn["output"]) for turn in transcript if "output" in turn
        )
        if output_tokens <= 0:
            efficiency = 0.0
        else:
            ratio = total_tokens / output_tokens
            efficiency = 1.0 if ratio < 2.0 else max(0.0, 1.0 - (ratio - 2.0) / 10.0)

        tools = [str(turn.get("tool", "")) for turn in transcript]
        redundancy_ratio = (count - len(set(tools))) / count

        latency_total = sum(_latency_ms(turn.get("latency_ms", 0)) for turn in transcript)
        average_latency = latency_total / count
        if average_latency <= 1000.0:
            speed = 1.0
        else:
            speed = max(0.0, 1.0 - (average_latency - 1000.0) / 10000.0)

        raw = (
            (correctness * 50.0)
            + (efficiency * 30.0)
            + (speed * 20.0)
            - (redundancy_ratio * 20.0)
        )
        return max(0.0, min(100.0, raw))
