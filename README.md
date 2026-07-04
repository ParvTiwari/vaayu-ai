# 🌫️ Vaayu AI

**A multi-agent air-quality intelligence system that turns raw monitoring-station
data into forecasts, source attribution, enforcement priorities, and multilingual
citizen advisories — with every high-stakes decision made by an auditable,
deterministic rule, not a black-box LLM.**

_ET AI Hackathon 2026 · Problem Statement 5 (Urban Air Quality)._

India records ~1.67 million premature deaths a year from air pollution, yet a CAG
audit found only **31% of 900+ CAAQMS stations have actionable response protocols**.
Vaayu AI is the missing "actionable response" layer: it reads the same station feeds
and produces forecasts, attribution, prioritized enforcement, and citizen guidance.

---

## What it does

| Feature | How | Auditable? |
|---|---|---|
| **24/48/72h AQI forecast** | XGBoost per horizon; beats a persistence baseline by **16.6–19.8% RMSE** | ML, benchmarked vs. baseline |
| **Source attribution** | Deterministic traffic / industrial / fire scoring over OSM + NASA FIRMS | ✅ no LLM — documented formula |
| **Enforcement priorities** | Zones ranked by `risk × vulnerability` (AQI severity × nearby hospitals/schools) | ✅ no LLM — documented formula |
| **Citizen advisory** | Hardcoded CPCB health-guidance table, LLM only rephrases/localizes | ✅ health facts never LLM-generated |
| **AQI heatmap** | Inverse-distance-weighting interpolation across the city grid | ✅ deterministic |
| **Multilingual + voice** | English / Hindi / Kannada, optional gTTS audio | — |
| **One-call orchestration** | LangGraph `StateGraph` routes a free-text query through the right agents | ✅ keyword router, no LLM |

Every advisory returns `sources_cited` (station, timestamp, AQI, category) so any
claim traces back to a real reading.

## Architecture at a glance

```
                         ┌──────────────────────────────┐
   POST /query  ───────▶ │      Orchestrator Agent      │  keyword intent router
   {city, lat, lon,      │   (detects query_type,       │  (deterministic, no LLM)
    lang, query}         │    normalizes language)      │
                         └───────────────┬──────────────┘
                                         │  conditional routing on query_type
        ┌────────────────┬───────────────┼────────────────┬────────────────┐
        ▼                ▼               ▼                 ▼                │
  ┌───────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐         │
  │ Forecast  │   │Attribution │   │Enforcement │   │  Advisory  │         │
  │  Agent    │   │   Agent    │   │   Agent    │   │   Agent    │         │
  │ (XGBoost) │   │  (rules)   │   │  (rules)   │   │(CPCB+LLM)  │         │
  └─────┬─────┘   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘         │
        │   "full": forecast → attribution → advisory (fed forward)        │
        └────────────────┴───────────────┬────────────────┴───────────────┘
                                          ▼
                              ┌───────────────────────┐
                              │     Output Agent       │  assembles final_response
                              │  (chat text + map      │  { chat:{…}, map:{…},
                              │   layers, no LLM)      │    sources_cited:{…} }
                              └───────────┬────────────┘
                                          ▼
                                        (END)

  Data layer:  CPCB/OpenAQ (AQI) · Open-Meteo (weather) · NASA FIRMS (fire) · OSM/Overpass (geo)
               → data/db/*.parquet + city_layers/*.geojson  → models/checkpoints/*.json
```

Full diagram, state schema, and error-handling tables: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Quick start

```bash
# 1. Environment (Python 3.9+; on macOS XGBoost needs OpenMP: `brew install libomp`)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add CPCB_API_KEY, OPENAQ_API_KEY, FIRMS_API_KEY, LLM_API_KEY (all optional — see below)

# 2. Ingest data → data/db/ (slow first time; caches everything. Skippable if data/db/ is populated)
python data/ingest/run_ingestion.py          # VAAYU_CITIES=Delhi to restrict; VAAYU_HISTORY_DAYS to bound

# 3. Train the forecast model → models/checkpoints/ + models/BENCHMARKS.md
python models/train_forecast_model.py

# 4. Run the API and the UI (two terminals)
uvicorn main:app --reload                     # http://localhost:8000  (docs at /docs)
streamlit run ui/app.py                       # http://localhost:8501

# 5. Tests
pytest -q                                     # 55 tests
```

**Keys are optional and the system degrades gracefully:** no `OPENAQ_API_KEY` → no
historical AQI; no `FIRMS_API_KEY` → fire feature is 0; no `LLM_API_KEY` → advisories
use the deterministic CPCB guidance (still fully localized for Hindi). See the
error-handling table in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Cities & languages supported

| City | AQI + weather + forecast | OSM layers (attribution / enforcement) | Notes |
|---|---|---|---|
| **Delhi** | ✅ 24 stations, 2016–2026, model trained | ✅ 3,543 road cells · 382 industrial · 3,360 POIs | Primary pilot — end-to-end |
| **Bengaluru** | ⏳ AQI backfill pending | ✅ 1,801 road cells · 1,285 industrial · 4,675 POIs | Attribution/enforcement live |
| **Indore** | ⏳ AQI backfill pending | ✅ 1,145 road cells · 51 industrial · 1,524 POIs | Attribution/enforcement live |

| Language | Deterministic CPCB guidance | LLM localization | Voice (gTTS) |
|---|---|---|---|
| English (`en`) | ✅ | ✅ | ✅ |
| Hindi (`hi`) | ✅ | ✅ | ✅ |
| Kannada (`kn`) | via English fallback | ✅ | ✅ |

## Tech stack

| Layer | Tools |
|---|---|
| **API** | FastAPI · Uvicorn · Pydantic |
| **Agent orchestration** | LangGraph (`StateGraph`) · langchain-core |
| **Forecast ML** | XGBoost · scikit-learn · pandas · numpy · pyarrow |
| **Geospatial** | custom numpy (haversine, IDW, shoelace) · hand-written GeoJSON (no geopandas) |
| **LLM (advisory narration)** | Anthropic Claude (`claude-opus-4-8`) via `LLM_API_KEY` — provider-swappable behind `call_llm()` |
| **Voice** | gTTS |
| **Frontend** | Streamlit · Folium · streamlit-folium · Plotly |
| **Data sources** | CPCB (`data.gov.in`) / OpenAQ v3 · Open-Meteo · NASA FIRMS · OpenStreetMap / Overpass |
| **Testing** | pytest (55 tests: unit, API, forecast regression, data integrity) |

## Repository layout

```
data/ingest/       CPCB/OpenAQ, weather, FIRMS, Overpass ingestion  (+ README.md)
data/db/           parquet datasets + city_layers geojson + raw_cache  (gitignored)
models/            train_forecast_model.py · checkpoints/ · BENCHMARKS.md
agents/            orchestrator, forecast, attribution, enforcement, advisory, output, graph
main.py            FastAPI app (all routes)
ui/app.py          Streamlit demo frontend
tests/test_all.py  full pytest suite
docs/              ARCHITECTURE.md · IMPACT_MODEL.md · BENCHMARKS.md
```

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — agents, state schema, tool integrations, error handling.
- **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** — accuracy vs. baseline, data coverage, measured latency, language coverage.
- **[docs/IMPACT_MODEL.md](docs/IMPACT_MODEL.md)** — quantified impact with explicit assumptions + sensitivity analysis.
- **[data/ingest/README.md](data/ingest/README.md)** — data sources, API keys, ingestion reliability notes.
