"""
Output Agent — assembles whichever sub-agent results were computed
(forecast / attribution / enforcement / advisory) into one final_response
shaped for both the map UI and the chat UI.

Not yet implemented — scaffold only.
"""
from __future__ import annotations

from typing import Any


def run_output(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: assemble the final response for the UI. TODO: implement."""
    raise NotImplementedError("run_output is not yet implemented")


__all__ = ["run_output"]
