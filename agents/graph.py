"""
LangGraph wiring for the Vaayu AI multi-agent pipeline.

Follows the AgriBloom/Creda pattern: a single StateGraph over a shared typed
state (`CityAirState`), an entry-point router (`orchestrator`), conditional
routing to the deterministic/ML sub-agents, and a terminal synthesizer
(`output`) that assembles `final_response` before END.

Topology
--------
    orchestrator                      (entry — sets query_type, normalizes lang)
        │  (conditional on query_type)
        ├── forecast ──┐
        │   (if "full") ├── attribution ──┐
        │               │   (if "full")    ├── advisory ── output ── END
        ├── attribution ┘ (if "attribution") ── output
        ├── enforcement ───────────────────────  output
        └── advisory ──────────────────────────  output

    query_type routing:
      "forecast"    → forecast → output
      "attribution" → attribution → output
      "enforcement" → enforcement → output
      "advisory"    → advisory → output
      "full"        → forecast → attribution → advisory → output   (fed forward)

Every sub-agent writes its slice of the shared state; the graph continues even
if a slice errors (the output node reports whatever was produced).

CityAirState schema (shared typed state)
----------------------------------------
    city            str   — pilot city name (e.g. "Delhi")
    lat, lon        float — query point (optional if station_id given)
    station_id      str   — monitoring station id (optional if lat/lon given)
    lang            str   — 2-letter language code (en/hi/kn), normalized
    query           str   — free-text query (used for intent detection)
    query_type      str   — forecast | attribution | enforcement | advisory | full
    forecast        dict  — Forecast Agent output (24/48/72h + primary)
    attribution     dict  — Attribution Agent output (traffic/industrial/fire scores)
    priority_zones  list  — Enforcement Agent ranked zones
    advisory_text   str   — Advisory Agent localized advisory
    sources_cited   dict  — provenance for the advisory reading
    voice_output_path str — optional TTS audio path
    methodology_note str  — enforcement methodology note
    final_response  dict  — Output Agent assembled response (chat + map)
    status          str   — "ok" | "error"
    error           str   — error detail when status == "error"
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.advisory_agent import run_advisory
from agents.attribution_agent import run_attribution
from agents.enforcement_agent import run_enforcement
from agents.forecast_agent import run_forecast
from agents.orchestrator_agent import run_orchestrator
from agents.output_agent import run_output


class CityAirState(TypedDict, total=False):
    city: str
    lat: float
    lon: float
    station_id: str
    lang: str
    query: str
    query_type: str
    forecast: dict[str, Any]
    attribution: dict[str, Any]
    priority_zones: list[dict[str, Any]]
    advisory_text: str
    sources_cited: dict[str, Any]
    voice_output_path: str
    methodology_note: str
    final_response: dict[str, Any]
    status: str
    error: str


# --- routing functions -------------------------------------------------------


def _route_from_orchestrator(state: CityAirState) -> str:
    """First node for the chosen query_type ('full' starts at forecast)."""
    qt = state.get("query_type", "full")
    if qt in ("forecast", "full"):
        return "forecast"
    if qt in ("attribution", "enforcement", "advisory"):
        return qt
    return "forecast"


def _after_forecast(state: CityAirState) -> str:
    return "attribution" if state.get("query_type") == "full" else "output"


def _after_attribution(state: CityAirState) -> str:
    return "advisory" if state.get("query_type") == "full" else "output"


# --- graph builder -----------------------------------------------------------


def build_air_quality_graph():
    """Build and compile the Vaayu AI LangGraph StateGraph."""
    g = StateGraph(CityAirState)

    g.add_node("orchestrator", run_orchestrator)
    g.add_node("forecast", run_forecast)
    g.add_node("attribution", run_attribution)
    g.add_node("enforcement", run_enforcement)
    g.add_node("advisory", run_advisory)
    g.add_node("output", run_output)

    g.set_entry_point("orchestrator")
    g.add_conditional_edges(
        "orchestrator",
        _route_from_orchestrator,
        {"forecast": "forecast", "attribution": "attribution",
         "enforcement": "enforcement", "advisory": "advisory"},
    )
    # forecast → (attribution if full) else output
    g.add_conditional_edges("forecast", _after_forecast, {"attribution": "attribution", "output": "output"})
    # attribution → (advisory if full) else output
    g.add_conditional_edges("attribution", _after_attribution, {"advisory": "advisory", "output": "output"})
    g.add_edge("enforcement", "output")
    g.add_edge("advisory", "output")
    g.add_edge("output", END)

    return g.compile()


# Compile once at import so the API/UI reuse a single graph instance.
air_quality_graph = build_air_quality_graph()


def run_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    """Invoke the compiled graph and return the final state dict."""
    return air_quality_graph.invoke(state)


__all__ = ["CityAirState", "build_air_quality_graph", "air_quality_graph", "run_pipeline"]
