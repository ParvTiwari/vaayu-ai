# Vaayu AI — System Architecture

Vaayu AI is a multi-agent urban air-quality intelligence system. A single
**LangGraph `StateGraph`** (`agents/graph.py`) routes each request through a
deterministic entry-point orchestrator to one or more sub-agents, then a terminal
synthesizer assembles a UI-ready response. The design centers on **a shared typed
state, a keyword router, deterministic (auditable) scoring agents, and one ML
agent**, with the LLM confined to a narration role and never trusted with a
high-stakes decision.

## The six agents

| Agent | File | Role | LLM? |
|---|---|---|---|
| **Orchestrator** | `agents/orchestrator_agent.py` | Entry-point router — keyword intent classifier sets `query_type`, normalizes `lang` | No |
| **Forecast** | `agents/forecast_agent.py` | 24/48/72h AQI from trained XGBoost models, CPCB category mapping, persistence fallback | No (ML) |
| **Attribution** | `agents/attribution_agent.py` | Rule-based traffic / industrial / fire source scoring over OSM + FIRMS; IDW heatmap | No |
| **Enforcement** | `agents/enforcement_agent.py` | Deterministic zone ranking by `risk × vulnerability` | No |
| **Advisory** | `agents/advisory_agent.py` | CPCB health-guidance lookup (ground truth) + LLM localization; source-cited | LLM narration only |
| **Output** | `agents/output_agent.py` | Synthesizer — assembles `final_response` (chat + map) from whatever ran | No |

Only the Advisory agent calls an LLM, and only to **rephrase/localize** the fixed
CPCB guidance — never to compute AQI, scores, rankings, or forecasts.

## Multi-agent graph topology

```
                        ┌───────────────────────────────┐
   entry ──────────────▶│        Orchestrator           │  sets query_type, lang
                        └───────────────┬───────────────┘
                                        │  _route_from_orchestrator (conditional)
        ┌───────────────┬───────────────┼───────────────┬────────────────┐
        ▼               ▼               ▼               ▼                │
  ┌───────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
  │ Forecast  │  │Attribution │  │Enforcement │  │  Advisory  │          │
  └─────┬─────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘          │
        │ _after_forecast      │              │        │                 │
        │ (full → attribution) │ _after_attribution    │                 │
        │                      │ (full → advisory)     │                 │
        └───────────────┬──────┴───────────────┬───────┴─────────────────┘
                        ▼                       ▼
                  ┌─────────────────────────────────────┐
                  │             Output Agent            │  final_response
                  └──────────────────┬──────────────────┘
                                     ▼
                                   (END)
```

Routing by `query_type` (set by the orchestrator, either explicit or keyword-detected):

| `query_type` | Path |
|---|---|
| `forecast` | orchestrator → forecast → output |
| `attribution` | orchestrator → attribution → output |
| `enforcement` | orchestrator → enforcement → output |
| `advisory` | orchestrator → advisory → output |
| `full` | orchestrator → **forecast → attribution → advisory** → output |

In `full`, each agent feeds the next: the Forecast writes `state["forecast"]`, which
the Advisory consumes as its headline AQI; the Attribution result is surfaced to the
Advisory prompt as the likely dominant source. **The graph continues even if a
sub-agent errors** — the Output node reports whichever slices were produced (a city
with no ingested AQI yields attribution-only, not a crash).

## Shared state — `CityAirState` (`TypedDict, total=False`)

| Field | Type | Written by | Meaning |
|---|---|---|---|
| `city` | str | caller | Pilot city (e.g. `"Delhi"`) |
| `lat`, `lon` | float | caller | Query point (optional if `station_id` given) |
| `station_id` | str | caller | Monitoring station id (optional if lat/lon given) |
| `lang` | str | orchestrator | 2-letter code (`en`/`hi`/`kn`), normalized |
| `query` | str | caller | Free-text query (used for intent detection) |
| `query_type` | str | orchestrator | `forecast` \| `attribution` \| `enforcement` \| `advisory` \| `full` |
| `forecast` | dict | forecast | 24/48/72h predictions + headline (`aqi`, `category`, `horizons`, `model`) |
| `attribution` | dict | attribution | `traffic_score`, `industrial_score`, `fire_score`, `overall_source_estimate`, `confidence` |
| `priority_zones` | list | enforcement | Ranked zones (`priority_score` desc) |
| `advisory_text` | str | advisory | Localized citizen advisory |
| `sources_cited` | dict | advisory | Provenance (station, timestamp, AQI, guidance source) |
| `voice_output_path` | str | advisory | Optional gTTS audio path |
| `methodology_note` | str | enforcement | Formula description |
| `final_response` | dict | output | Assembled response (`chat` + `map` + `sources_cited`) |
| `status` | str | all | `ok` \| `error` |
| `error` | str | all | Detail when `status == error` |

`final_response` carries a `chat` block (summary text + advisory, for the chat pane)
and a `map` block (forecast marker, attribution scores, priority-zone markers, a
heatmap hint) for map rendering.

## Tool integrations

| Agent / layer | Tool | Purpose | Fallback on failure |
|---|---|---|---|
| Forecast | XGBoost models (`models/checkpoints/forecast_*.json`) | 24/48/72h AQI prediction | **Persistence** — carry latest observed AQI forward (`model="persistence"`) if models/features missing |
| Forecast | `unified_history.parquet` | Feature source (lags, rolling, weather, fire, calendar) | `status="error"` if the station has no history |
| Attribution | OSM `city_layers/*.geojson` (Overpass) | Road-density grid + industrial polygons | Score 0; `layers_available=false` recorded |
| Attribution | `fire_daily.parquet` (NASA FIRMS) | Stubble-burning fire score | 0 when empty or outside Oct–Nov season |
| Attribution | IDW over `unified_history` | AQI heatmap interpolation | Empty grid → HTTP 404 on `/heatmap` |
| Enforcement | OSM vulnerability POIs + latest AQI | `risk × vulnerability` ranking | Station skipped if it has no AQI reading |
| Advisory | `HEALTH_GUIDANCE` table (CPCB, hardcoded) | Ground-truth health facts | Always available (in-code constant) |
| Advisory | LLM via `LLM_API_KEY` (Groq / Anthropic / OpenAI-compatible, auto-detected) | Rephrase/localize the guidance | **Deterministic** CPCB guidance text (`narration="deterministic_fallback"`) |
| Advisory | gTTS | Voice audio | `voice_output_path=None` if gTTS missing / fails / `TTS_ENABLED` unset |
| Ingestion | CPCB `data.gov.in` | Live AQI snapshot | OpenAQ `latest` fallback; ret/timeout backoff |
| Ingestion | OpenAQ v3 | Historical AQI | Empty frame if no `OPENAQ_API_KEY` |
| Ingestion | Open-Meteo archive | Weather features | Retry without `boundary_layer_height` on 400; null weather on failure |
| Ingestion | NASA FIRMS area/CSV | Active-fire detections | Empty frame if no `FIRMS_API_KEY` |
| Ingestion | Overpass API | OSM road/industrial/POI layers | User-Agent fix → Retry-After retries → shrink-bbox fallback → cached/empty |
| HTTP core | `requests` + on-disk cache (`_http_utils`) | All external calls | On-disk cache, exponential backoff, `VAAYU_CACHE_ONLY` offline mode |

## Error-handling summary

Every external dependency has an explicit degradation path; nothing in the request
path raises to the caller.

| Scenario | Trigger | Handling | User-visible result |
|---|---|---|---|
| **CPCB API down / slow** | Read timeout after 3 retries | Log, fall back to OpenAQ `latest` | Live AQI from OpenAQ, or empty (no crash) |
| **OpenAQ deep-pagination timeout** | A later `/hours` page 408s | Stop paginating; keep pages already fetched | Partial history, no crash |
| **No `OPENAQ_API_KEY`** | Key unset | Skip OpenAQ with a clear warning | No historical AQI ingested |
| **No `FIRMS_API_KEY`** | Key unset | Skip fire ingestion | `nearby_fire_count = 0`, `fire_score = 0` (documented) |
| **Overpass 406 Not Acceptable** | Bare `requests` User-Agent rejected | Send descriptive User-Agent | Layers fetch normally |
| **Overpass 429 / 504** | Rate limit / gateway timeout | Retry-After-aware retries → shrink-bbox (¼-area) → cached/empty | Layer status `ok`/`partial`/`fallback`/`failed` in `_meta` |
| **Missing / thin station history (forecast)** | < ~200 hourly rows, or model absent | **Persistence** forecast (`forecast(t+h) = latest observed AQI`) | Forecast still returned, `model="persistence"` |
| **Unknown / missing city or point** | Validation in the agent | `status="error"` + `error` | HTTP 400 (single routes) / graceful skip (`/query`) |
| **Out-of-range coordinates** | `lat`/`lon` outside ±90 / ±180 | Pydantic / Query bounds | HTTP 422 |
| **Low / zero attribution signal** | All three scores 0 | `overall_source_estimate="indeterminate"`, `confidence="low"` | Honest "indeterminate", no guess |
| **Over-confident attribution** | Correlational heuristic by design | Confidence hard-capped at `"medium"` | Never claims causal certainty |
| **No AQI available for advisory** | No forecast **and** no observed reading | `status="error"` | HTTP 400 |
| **No `LLM_API_KEY` / LLM call fails** | Key unset, or network/auth/timeout error | Deterministic CPCB guidance text | Advisory still returned; `narration="deterministic_fallback"` |
| **TTS unavailable** | `TTS_ENABLED` unset, gTTS missing, or synth error | `voice_output_path=None` | Advisory text without audio |
| **Sub-agent error mid-`/query`** | Any agent sets `status="error"` | Graph continues; Output reports components that ran | Partial `final_response` (`status="empty"` if none ran) |
| **AQI beyond the CPCB scale** | `estimate_aqi` on extreme/garbage PM | Capped at the official ceiling of 500 | AQI always within 0–500 |
| **Backend unreachable (UI)** | `requests` connection error | Red banner + `st.stop()` | Clear "start `uvicorn`" message, no traceback |

## Data dependencies

- `data/db/unified_history.parquet` — AQI + weather per station (Forecast, Advisory, Enforcement, IDW).
- `data/db/fire_daily.parquet` — daily fires-near-station (Forecast feature, Attribution).
- `data/db/city_layers/{city}_layers.geojson` — OSM road grid / industrial / POIs (Attribution, Enforcement).
- `models/checkpoints/forecast_{24,48,72}h.json` + `forecast_metadata.json` — trained models + feature list.

See `data/ingest/README.md` for ingestion detail and `docs/BENCHMARKS.md` for the
model card, coverage, and measured latency.
