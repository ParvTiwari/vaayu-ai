"""
Citizen Advisory Agent — wraps a deterministic CPCB/WHO health-guidance
lookup table with an LLM narrative layer for multilingual, voice-ready
citizen advisories. The LLM only rephrases/localizes; it never invents
AQI numbers or health claims.

Not yet implemented — scaffold only.
"""
from __future__ import annotations

from typing import Any

# CPCB health-guidance text per AQI category, per language.
# Deterministic — sourced from official CPCB guidance, not LLM-generated.
HEALTH_GUIDANCE: dict[str, dict[str, str]] = {
    "en": {},
    "hi": {},
}


def run_advisory(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: produce a localized, source-cited citizen advisory. TODO: implement."""
    raise NotImplementedError("run_advisory is not yet implemented")


__all__ = ["run_advisory", "HEALTH_GUIDANCE"]
