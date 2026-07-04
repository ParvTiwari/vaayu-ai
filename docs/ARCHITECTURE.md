# Vaayu AI — System Architecture

Vaayu AI is a multi-agent urban air-quality intelligence system. A single
**LangGraph `StateGraph`** (`agents/graph.py`) routes each request through a
deterministic entry-point orchestrator to one or more sub-agents, then a
terminal output node assembles a UI-ready response. The design deliberately
mirrors the AgriBloom/Creda hackathon pattern: **a shared typed state, a
keyword router, deterministic (auditable) scoring agents, and one ML agent**,
with the LLM confined to a narration role.

## Agents

| Agent | File | Role | LLM? |
|---|---|---|---|
| **Orchestrator** | `agents/orchestrator_agent.py` | Entry-point router — detects `query_type` (keyword intent classifier) and normalizes `lang` | No |
| **Forecast** | `agents/forecast_agent.py` | 24/48/72h AQI from the trained XGBoost models (`models/checkpoints/`), CPCB category mapping, persistence fallback | No (ML) |
| **Attribution** | `agents/attribution_agent.py` | Rule-based source scoring (traffic / industrial / fire) over OSM + FIRMS layers + IDW heatmap | No |
| **Enforcement** | `agents/enforcement_agent.py` | Deterministic zone ranking by `risk × vulnerability` | No |
| **Advisory** | `agents/advisory_agent.py` | CPCB health-guidance lookup (ground truth) + LLM localization; source-cited | LLM narration only |
| **Output** | `agents/output_agent.py` | Synthesizer — assembles `final_response` (chat + map) from whatever ran | No |

Only the Advisory agent calls an LLM, and only to **rephrase/localize** the
fixed CPCB guidance — never to compute AQI, scores, rankings, or forecasts.

## Graph topology

```
orchestrator                       (entry — sets query_type, normalizes lang)
    │  add_conditional_edges(_route_from_orchestrator)
    ├── forecast ──add_conditional_edges(_after_forecast)──┐
    │       │                                              │
    │       └─(full)→ attribution ─add_conditional_edges(_after_attribution)─┐
    │                     │                                                  │
    ├── attribution ──────┘ (attribution-only)→ output          (full)→ advisory
    ├── enforcement ─────────────────────────── output                   │
    └── advisory ─────────────────────────────── output ←────────────────┘
                                                    │
                                                   END
```

Routing by `query_type`:

| `query_type` | Path |
|---|---|
| `forecast` | orchestrator → forecast → output |
| `attribution` | orchestrator → attribution → output |
| `enforcement` | orchestrator → enforcement → output |
| `advisory` | orchestrator → advisory → output |
| `full` | orchestrator → **forecast → attribution → advisory** → output |

In the `full` path each agent's output is fed forward: the Forecast writes
`state["forecast"]`, which the Advisory agent consumes as its headline AQI; the
Attribution result is surfaced to the Advisory prompt as the likely dominant
source. The graph **continues even if a sub-agent errors** — the Output node
reports whichever slices were produced (e.g. a city with no ingested AQI history
yields attribution-only).

## Shared state — `CityAirState` (TypedDict, `total=False`)

| Field | Type | Written by | Meaning |
|---|---|---|---|
| `city` | str | caller | Pilot city (e.g. `"Delhi"`) |
| `lat`, `lon` | float | caller | Query point (optional if `station_id` given) |
| `station_id` | str | caller | Monitoring station id (optional if lat/lon given) |
| `lang` | str | orchestrator | 2-letter language code (`en`/`hi`/`kn`), normalized |
| `query` | str | caller | Free-text query (used for intent detection) |
| `query_type` | str | orchestrator | `forecast` \| `attribution` \| `enforcement` \| `advisory` \| `full` |
| `forecast` | dict | forecast | 24/48/72h predictions + headline (`aqi`, `category`, `horizons`, `model`) |
| `attribution` | dict | attribution | `traffic_score`, `industrial_score`, `fire_score`, `overall_source_estimate`, `confidence` |
| `priority_zones` | list | enforcement | Ranked zones (`priority_score` desc) |
| `advisory_text` | str | advisory | Localized citizen advisory |
| `sources_cited` | dict | advisory | Provenance (station, timestamp, AQI, guidance source) |
| `voice_output_path` | str | advisory | Optional gTTS audio path |
| `methodology_note` | str | enforcement | Formula description |
| `final_response` | dict | output | Assembled response (see below) |
| `status` | str | all | `ok` \| `error` |
| `error` | str | all | Error detail when `status == error` |

### `final_response` shape (Output agent)

```jsonc
{
  "status": "ok",
  "query_type": "full",
  "city": "Delhi",
  "language": "en",
  "location": { "lat": 28.57, "lon": 77.07, "station_id": null },
  "components_run": ["forecast", "attribution", "advisory"],
  "chat": {                              // for the chat pane
    "summary": "Forecast (+24h): ... Likely dominant source: ... <advisory>",
    "advisory": "<localized advisory>",
    "lines": ["...", "...", "..."]
  },
  "map": {                               // for map rendering
    "forecast": { "type": "station_forecast", "lat","lon", "aqi","category","horizons" },
    "attribution": { "type": "source_scores", "point", "traffic_score", ... },
    "priority_zones": [ { "type": "priority_zone", "lat","lon","priority_score", ... } ],
    "heatmap_hint": { "endpoint": "/heatmap/Delhi", "timestamp": "..." }
  },
  "sources_cited": { ... }
}
```

## API routes (`main.py`, FastAPI)

| Method / Route | Backed by | Purpose |
|---|---|---|
| `GET /` | — | Health check |
| `POST /query` | `graph.run_pipeline` | **Full pipeline** — `{city, lat, lon, station_id, lang, query_type, query}` → `final_response` |
| `GET /attribution/{city}?lat=&lon=` | Attribution agent | Source scoring at a point |
| `GET /heatmap/{city}?timestamp=` | Attribution IDW | Interpolated AQI grid (GeoJSON) |
| `GET /enforcement-priorities/{city}` | Enforcement agent | Ranked priority zones |
| `POST /advisory` | Advisory agent | Localized, source-cited advisory |

`POST /query` is the single entry point that exercises the whole graph; the
other routes expose individual agents directly (useful for the map UI, which
calls `/heatmap` and `/enforcement-priorities` for dedicated layers).

## Error handling

| Condition | Behavior |
|---|---|
| Unknown / missing city or point | Agent sets `status="error"` + `error`; graph continues; Output omits that component |
| No ingested AQI history for a city | Forecast/Advisory degrade or skip; Attribution/Enforcement still run on OSM/FIRMS layers |
| Forecast model unavailable / thin history | Forecast falls back to **persistence** (`forecast(t+h) = latest observed AQI`) |
| No `FIRMS_API_KEY` at ingestion | `fire_score` / `nearby_fire_count` are 0 (documented), other signals unaffected |
| No `LLM_API_KEY` or LLM call fails | Advisory falls back to the **deterministic** CPCB guidance text |
| Overpass/upstream API flaky | Cached layers reused; `_meta.status` records `ok`/`partial`/`fallback`/`failed` |

## Data dependencies

- `data/db/unified_history.parquet` — AQI + weather per station (Forecast, Advisory, Enforcement, IDW).
- `data/db/fire_daily.parquet` — daily fires-near-station (Forecast feature, Attribution).
- `data/db/city_layers/{city}_layers.geojson` — OSM road grid / industrial / POIs (Attribution, Enforcement).
- `models/checkpoints/forecast_{24,48,72}h.json` + `forecast_metadata.json` — trained models + feature list.

See `data/ingest/README.md` for ingestion and `models/BENCHMARKS.md` for the
forecast model card.
