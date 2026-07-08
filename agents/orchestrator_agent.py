"""
Orchestrator Agent — the LangGraph entry-point router.

run_orchestrator normalizes the request and decides which downstream agent(s)
should run. If `query_type` is supplied explicitly it is respected; otherwise a
simple keyword intent classifier reads
`state["query"]` and picks one of forecast / attribution / enforcement /
advisory / full. Language is normalized to a 2-letter code (default 'en').

No LLM is used for routing — it is a deterministic, auditable keyword map.
"""
from __future__ import annotations

from typing import Any

VALID_QUERY_TYPES = {"forecast", "attribution", "enforcement", "advisory", "full"}
DEFAULT_QUERY_TYPE = "full"

# Ordered so the most specific intents win when several keywords co-occur.
# Advisory is checked before forecast on purpose: a safety question like
# "is it safe to jog tomorrow?" mentions "tomorrow" but is fundamentally an
# advisory request, so the health/safety words should win over "tomorrow".
_INTENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("enforcement", ("enforce", "priority", "prioriti", "inspect", "hotspot", "worst zone",
                     "worst-affected", "action plan", "rank")),
    ("attribution", ("source", "cause", "blame", "attribut", "traffic", "industrial",
                     "factory", "stubble", "responsible", "contribut", "whose fault", "who is to blame")),
    ("advisory", ("advisory", "advice", "should i", "safe to", "is it safe", "safe for",
                  "health", "mask", "go out", "jog", "run outside", "exercise", "children",
                  "elderly", "recommend", "precaution")),
    ("forecast", ("forecast", "predict", "tomorrow", "next 24", "next 48", "next 72",
                  "will it", "expected", "outlook", "coming days", "day after")),
    ("full", ("full", "everything", "complete", "overall", "all of", "summary", "report",
              "brief me")),
]

_LANG_ALIASES = {
    "english": "en", "en": "en", "eng": "en",
    "hindi": "hi", "hi": "hi", "hin": "hi",
    "kannada": "kn", "kn": "kn", "kan": "kn",
}


def _detect_intent(query: str) -> str:
    q = (query or "").lower()
    if not q.strip():
        return DEFAULT_QUERY_TYPE
    for intent, keywords in _INTENT_KEYWORDS:
        if any(k in q for k in keywords):
            return intent
    return DEFAULT_QUERY_TYPE


def _normalize_lang(lang: Any) -> str:
    if not lang:
        return "en"
    key = str(lang).strip().lower()
    return _LANG_ALIASES.get(key, key if len(key) == 2 else "en")


def run_orchestrator(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize inputs and set state['query_type']. See module docstring."""
    qt = state.get("query_type")
    if qt not in VALID_QUERY_TYPES:
        qt = _detect_intent(state.get("query", ""))
    state["query_type"] = qt
    state["lang"] = _normalize_lang(state.get("lang"))
    state.setdefault("status", "ok")
    return state


__all__ = ["run_orchestrator", "VALID_QUERY_TYPES"]
