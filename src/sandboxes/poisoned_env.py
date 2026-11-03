"""Reject tool output that is not a weather observation.

Untrusted tool strings are parsed with ``json.loads`` and validated against
``WeatherObservation``. Extra fields, non-objects, non-finite numbers, oversized
payloads, and invalid JSON become a rejection object. The original string is
never evaluated or executed.
"""

from __future__ import annotations

import json
import math
from typing import NoReturn

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

MAX_TOOL_OUTPUT_CHARS = 4096


class WeatherObservation(BaseModel):
    """Temperature report the agent is allowed to read."""

    model_config = ConfigDict(extra="forbid")

    temperature: float
    unit: str = "C"

    @field_validator("temperature")
    @classmethod
    def _finite_temperature(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("temperature must be finite")
        return value

    @field_validator("unit")
    @classmethod
    def _known_unit(cls, value: str) -> str:
        if value not in {"C", "F", "K"}:
            raise ValueError("unit must be C, F, or K")
        return value


def _rejected(reason: str) -> str:
    return json.dumps({"rejected": True, "reason": reason})


def _reject_json_constant(_value: str) -> NoReturn:
    raise ValueError("non-finite json constant")


class PoisonedEnvAdapter(BaseModel):
    """Return schema-valid weather JSON, or a rejection object."""

    model_config = ConfigDict(extra="forbid")

    def __call__(self, tool_output: str) -> str:
        if len(tool_output) > MAX_TOOL_OUTPUT_CHARS:
            return _rejected("too large")
        try:
            parsed: object = json.loads(tool_output, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError, TypeError):
            return _rejected("invalid json")
        if not isinstance(parsed, dict):
            return _rejected("expected object")
        try:
            observation = WeatherObservation.model_validate(parsed)
        except ValidationError:
            return _rejected("schema mismatch")
        return observation.model_dump_json()
