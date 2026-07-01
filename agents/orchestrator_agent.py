"""
Orchestrator Agent — routes incoming queries to the right downstream agent(s).

Will detect query_type (forecast / attribution / enforcement / advisory / full),
normalize city/lang inputs, and hand off to the LangGraph pipeline defined in
agents/graph.py.

Not yet implemented — scaffold only.
"""
from __future__ import annotations

from typing import Any


def run_orchestrator(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: routes a request based on detected intent. TODO: implement."""
    raise NotImplementedError("run_orchestrator is not yet implemented")


__all__ = ["run_orchestrator"]
