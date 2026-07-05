"""
Output Agent — the LangGraph synthesizer/terminal node.

run_output assembles whichever sub-agent results were computed
(forecast / attribution / enforcement / advisory) into a single `final_response`
dict shaped for BOTH surfaces of the UI:
  - `chat`: human-readable summary text (+ the localized advisory) for the chat pane
  - `map`:  structured layers (forecast marker, attribution scores, priority-zone
            markers, a heatmap hint) for map rendering

No LLM here — it is a deterministic assembler of already-computed results.
"""
from __future__ import annotations

from typing import Any


def _forecast_sentence(f: dict) -> str:
    cat = (f.get("category") or "unknown").replace("_", " ")
    h = f.get("horizon_hours", 24)
    ref = f.get("reference_aqi")
    trend = ""
    if ref is not None and f.get("aqi") is not None:
        trend = " (improving)" if f["aqi"] < ref else (" (worsening)" if f["aqi"] > ref else " (steady)")
    return f"Forecast (+{h}h): AQI ~{f.get('aqi')} — {cat}{trend} [{f.get('model')} model]."


def _attribution_sentence(a: dict) -> str:
    primary = a.get("overall_source_estimate", "indeterminate")
    conf = a.get("confidence", "low")
    scores = f"traffic {a.get('traffic_score')}, industrial {a.get('industrial_score')}, fire {a.get('fire_score')}"
    return f"Likely dominant source: {primary} ({conf} confidence) — scores: {scores}."


def _enforcement_sentence(zones: list) -> str:
    if not zones:
        return "No priority zones computed."
    top = zones[0]
    return (f"{len(zones)} zones ranked; top priority: {top.get('station_id')} "
            f"(priority {top.get('priority_score')}, AQI {top.get('current_aqi')}).")


def run_output(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble final_response from whatever the pipeline produced."""
    forecast = state.get("forecast")
    attribution = state.get("attribution")
    priority_zones = state.get("priority_zones")
    advisory_text = state.get("advisory_text")
    sources_cited = state.get("sources_cited")

    components: list[str] = []
    chat_lines: list[str] = []
    map_layers: dict[str, Any] = {}

    if forecast:
        components.append("forecast")
        chat_lines.append(_forecast_sentence(forecast))
        map_layers["forecast"] = {
            "type": "station_forecast",
            "lat": forecast.get("lat"),
            "lon": forecast.get("lon"),
            "station_id": forecast.get("station_id"),
            "station_name": forecast.get("station_name"),
            "aqi": forecast.get("aqi"),
            "category": forecast.get("category"),
            "horizons": forecast.get("horizons"),
        }

    if attribution:
        components.append("attribution")
        chat_lines.append(_attribution_sentence(attribution))
        map_layers["attribution"] = {
            "type": "source_scores",
            "point": attribution.get("details", {}).get("point"),
            "traffic_score": attribution.get("traffic_score"),
            "industrial_score": attribution.get("industrial_score"),
            "fire_score": attribution.get("fire_score"),
            "overall_source_estimate": attribution.get("overall_source_estimate"),
            "confidence": attribution.get("confidence"),
        }

    if priority_zones is not None:
        components.append("enforcement")
        chat_lines.append(_enforcement_sentence(priority_zones))
        map_layers["priority_zones"] = [
            {"type": "priority_zone", "lat": z.get("lat"), "lon": z.get("lon"),
             "station_id": z.get("station_id"), "priority_score": z.get("priority_score"),
             "risk_score": z.get("risk_score"), "current_aqi": z.get("current_aqi"),
             "nearby_vulnerable_pois": z.get("nearby_vulnerable_pois")}
            for z in priority_zones
        ]

    if advisory_text:
        components.append("advisory")
        chat_lines.append(advisory_text)

    # A hint the UI can use to request the interpolated AQI heatmap layer.
    if state.get("city"):
        ts = (forecast or {}).get("reference_timestamp") or (sources_cited or {}).get("reading_timestamp")
        map_layers["heatmap_hint"] = {"endpoint": f"/heatmap/{state['city']}", "timestamp": ts}

    final_response = {
        "status": "ok" if components else "empty",
        "query_type": state.get("query_type"),
        "city": state.get("city"),
        "language": state.get("lang", "en"),
        "location": {
            "lat": state.get("lat"),
            "lon": state.get("lon"),
            "station_id": state.get("station_id"),
        },
        "components_run": components,
        "chat": {
            "summary": " ".join(chat_lines) if chat_lines else "No results were produced for this query.",
            "advisory": advisory_text,
            "lines": chat_lines,
        },
        "map": map_layers,
        "sources_cited": sources_cited,
        "voice_output_path": state.get("voice_output_path"),
    }
    state["final_response"] = final_response
    state["status"] = "ok"
    return state


__all__ = ["run_output"]
