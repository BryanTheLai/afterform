"""Shared ``GenerateContentConfig`` for product Gemini calls (KISS / DRY).

Thinking knobs live here only — stages pass stage-specific fields
(temperature, ``response_mime_type``, ``system_instruction``, …).
"""

from __future__ import annotations

from typing import Any

from google.genai import types

_THINKING = types.ThinkingConfig(
    thinking_budget=1024,
    include_thoughts=True,
)


def gemini_generate_config(**kwargs: Any) -> types.GenerateContentConfig:
    """Return config with thinking enabled; ``kwargs`` are merged as-is."""
    return types.GenerateContentConfig(
        thinking_config=_THINKING,
        **kwargs,
    )

