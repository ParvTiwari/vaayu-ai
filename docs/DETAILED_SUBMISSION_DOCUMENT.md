# Vaayu AI — Detailed Submission Document

**ET AI Hackathon 2.0 · Phase 2: Build Sprint · Problem Statement 5 (Urban Air Quality)**

---

## 1. Problem Statement

India records **~1.67 million premature deaths a year** from air pollution. The
government has built a large sensing network — **900+ CAAQMS monitoring
stations** — but a CAG audit found that **only 31% of them have any actionable
response protocol**. In other words: the data exists, but for roughly 7 in 10
stations, nobody converts a reading into a decision. There is no missing
sensor network — there is a missing **intelligence layer** that turns a raw
AQI number into a forecast, a cause, a priority, and an action.

**Vaayu AI is that layer.** It reads the same station feeds regulators already
have and produces:

1. A **24/48/72-hour AQI forecast**, so action happens before a spike, not after.
2. **Source attribution** (traffic / industrial / stubble-fire), so the right
   intervention gets chosen.
3. **Enforcement prioritization**, so limited inspection capacity goes to the
   highest-harm zones first.
4. A **citizen advisory**, in the citizen's own language, so individuals know
   what to do today.

## 2. Proposed Solution — Design Philosophy

The central design decision, made explicit throughout the system, is:
**machine learning forecasts, deterministic rules decide, and an LLM only
talks.**

Every number that could affect health guidance, an enforcement ranking, or a
"who's responsible" claim is produced by a documented formula or a trained,
benchmarked model — never invented by a language model. The LLM is used
exactly once in the whole pipeline (citizen-advisory narration/translation),
and even then it is instructed to rephrase a fixed CPCB guidance table, never
to generate a health fact. This matters for a civic-safety system: every claim
Vaayu makes is auditable back to a real station reading, a real formula, or a
benchmarked model — not a black box.

## 3. System Architecture

Vaayu AI is a **six-agent LangGraph pipeline** (`agents/graph.py`), entered
through a deterministic keyword-based orchestrator that classifies intent and
routes to one or more sub-agents:

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
  │  Agent    │   │   Agent    │   │   Agent    │   │(CPCB+LLM)  │         │
  │ (XGBoost) │   │  (rules)   │   │  (rules)   │   │            │         │
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

| Agent | Role | Uses an LLM? |
|---|---|---|
| **Orchestrator** | Keyword intent classifier; sets `query_type`, normalizes `lang` | No |
| **Forecast** | 24/48/72h AQI from a trained XGBoost model; persistence fallback | No (ML) |
| **Attribution** | Rule-based traffic/industrial/fire source scoring over OSM + NASA FIRMS; IDW heatmap | No |
| **Enforcement** | Deterministic zone ranking by `risk × vulnerability` | No |
| **Advisory** | CPCB health-guidance lookup (ground truth) + LLM localization | LLM narration only |
| **Output** | Synthesizer — assembles the final chat + map response | No |

**The graph degrades gracefully rather than crashing:** if a sub-agent errors
(e.g. a city has no ingested AQI), the Output node still returns whatever
slices did succeed. Every external dependency (CPCB, OpenAQ, Open-Meteo,
FIRMS, Overpass, the LLM provider, gTTS) has an explicit, documented fallback
path — full table in [docs/ARCHITECTURE.md](ARCHITECTURE.md).

## 4. Key Features

| Feature | How it works | Auditable? |
|---|---|---|
| 24/48/72h AQI forecast | XGBoost per horizon, beats a persistence baseline by 17.3–22.5% RMSE across all 3 pilot cities | ML, benchmarked |
| Source attribution | Deterministic traffic/industrial/fire scoring over OSM + NASA FIRMS | ✅ documented formula |
| Enforcement priorities | Zones ranked by `risk × vulnerability` (AQI severity × nearby hospitals/schools) | ✅ documented formula |
| Citizen advisory | Hardcoded CPCB health-guidance table; LLM only rephrases/localizes | ✅ health facts never LLM-generated |
| AQI heatmap | Inverse-distance-weighting interpolation across the city grid | ✅ deterministic |
| Multilingual + voice | English / Hindi / Kannada, optional gTTS audio | — |
| One-call orchestration | LangGraph `StateGraph` routes a free-text query through the right agents | ✅ keyword router, no LLM |

Every advisory response returns `sources_cited` (station, timestamp, AQI,
guidance source), so any claim traces back to a real reading.

## 5. Tech Stack

| Layer | Tools |
|---|---|
| API | FastAPI · Uvicorn · Pydantic |
| Agent orchestration | LangGraph (`StateGraph`) · langchain-core |
| Forecast ML | XGBoost · scikit-learn · pandas · numpy · pyarrow |
| Geospatial | Custom numpy (haversine, IDW, shoelace) · hand-written GeoJSON (no geopandas) |
| LLM (advisory narration only) | Provider-flexible via `LLM_API_KEY` — Groq (default), Anthropic Claude, or any OpenAI-compatible endpoint |
| Voice | gTTS |
| Frontend | Streamlit · Folium · streamlit-folium · Plotly |
| Data sources | CPCB (`data.gov.in`) / OpenAQ v3 · Open-Meteo · NASA FIRMS · OpenStreetMap / Overpass |
| Testing | pytest — 55 tests: unit, API, forecast regression, data integrity |

## 6. Results — Measured, Not Estimated

The forecast model is benchmarked against a persistence baseline
(`forecast(t+h) = actual(t)`) on a strict time-based split (test is always
later than train — no leakage), over **385,509 hourly feature rows from 46
stations across Delhi, Bengaluru, and Indore, 2016–2026**:

| City | Horizon | Persistence RMSE | Vaayu Model RMSE | Improvement |
|---|---|---:|---:|:--:|
| Delhi | 24h | 109.49 | **87.56** | **+20.0%** |
| Delhi | 48h | 118.28 | **91.63** | **+22.5%** |
| Delhi | 72h | 118.95 | **94.25** | **+20.8%** |
| Bengaluru | 24h | 44.78 | **35.86** | **+19.9%** |
| Bengaluru | 48h | 44.90 | **36.28** | **+19.2%** |
| Bengaluru | 72h | 46.64 | **37.22** | **+20.2%** |
| Indore | 24h | 31.87 | **26.35** | **+17.3%** |
| Indore | 48h | 34.13 | **27.41** | **+19.7%** |
| Indore | 72h | 34.58 | **28.19** | **+18.5%** |

**The model beats persistence at every horizon in all three pilot cities** — a
17.3–22.5% RMSE reduction across the full matrix. At Delhi 24h this cuts mean
forecast error by ~22 AQI points versus "assume tomorrow looks like today" —
precisely the days when persistence fails hardest (the onset/clearing of a
pollution episode) are the days advance warning matters most.

**End-to-end latency** (warm cache, local Uvicorn, measured pre-3-city-merge
against Delhi): every route responds in under a second, from `/advisory` at
~2ms to the heaviest route, `/enforcement-priorities`, at ~640ms for all 24
Delhi stations. Full latency table, feature importances, and data-coverage
numbers (OSM: 3,543 Delhi road cells / 382 industrial zones / 3,360
vulnerability POIs, plus Bengaluru and Indore layers) are in
[docs/BENCHMARKS.md](BENCHMARKS.md) and the auto-generated model card,
[models/BENCHMARKS.md](../models/BENCHMARKS.md).

**Test suite:** `pytest -q` → 55 tests passing across forecast regression
(against the 3-city pooled model), attribution scoring, enforcement ranking,
advisory (all 3 languages, verified against a live Groq LLM key), API
endpoints, and data-integrity checks (no duplicate station+timestamp rows, all
AQI in [0, 500], ≥5 stations per ingested city).

## 7. Impact Assessment

Full assumption-by-assumption model in
[docs/IMPACT_MODEL.md](IMPACT_MODEL.md); headline figures:

- **Year 1 (Delhi, 5% adoption, conservative):** ~360 avoided acute
  pollution-related hospital/ER visits, ≈ **₹10.8M / ~$130k**.
- **5-year cumulative (Delhi, adoption growing 5%→30%):** ~5,904 avoided
  events, ≈ **₹177M / ~$2.1M**.
- **Sensitivity is disclosed, not hidden:** the result is a product of five
  independent assumptions (reach, action rate, avoidance efficacy, warnable
  share, morbidity ratio) — halving any one halves the headline. The honest
  Year-1 range is **~180–1,620 avoided events**, not a single point estimate.
- **Beyond health arithmetic:** enforcement prioritization directs scarce
  inspection capacity at the 69% of stations with no protocol today; source
  attribution supports the *right* intervention (traffic vs. industrial vs.
  stubble) instead of blanket measures; vulnerability weighting explicitly
  protects zones near hospitals, schools, and elderly care.

## 8. Data Sources & Reliability

| Source | Used for | Status |
|---|---|---|
| CPCB `data.gov.in` + OpenAQ v3 | Live + historical AQI | 555,802 rows, 46 stations across Delhi (24), Bengaluru (18), Indore (5), 2016–2026 |
| Open-Meteo archive | Weather features | 300,349 rows joined within 1h of an AQI reading |
| NASA FIRMS (VIIRS) | Active-fire count (stubble burning) | Active across all 3 cities — 18,007 station-day rows, 111,750 raw unique detections; feature importance still low (~0.5%), not yet a proven forecast win on its own |
| OpenStreetMap / Overpass | Roads, industrial zones, vulnerability POIs | Live for Delhi, Bengaluru, Indore |

Every external call has a documented fallback (retry/backoff, on-disk cache,
`Retry-After`-aware Overpass handling, shrink-bbox degradation) — see the
error-handling table in [docs/ARCHITECTURE.md](ARCHITECTURE.md) and
[data/ingest/README.md](../data/ingest/README.md) for what actually broke
during real ingestion runs and how it was handled.

## 9. Honest Limitations (Disclosed, Not Hidden)

- **The fire feature is active but not yet proven to help on its own.** Real
  NASA FIRMS data now covers all 3 cities (18,007 station-day rows), but
  `nearby_fire_count`'s own XGBoost feature importance is low (~0.4–0.5%,
  ranked 17th–18th of 23 features at every horizon). The accuracy gain over
  the earlier Delhi-only, fire-inactive run is more plausibly explained by
  pooling three cities' training data than by the fire signal — we're
  disclosing that rather than claiming a win we haven't isolated.
- Forecast features use **observed-at-t weather**, not forecast weather — a
  mild, documented optimism; production should swap in Open-Meteo *forecast*
  weather at t+h.
- AQI is **approximated from PM2.5/PM10 sub-indices only** (no NO₂/SO₂/O₃/CO),
  inherited from what CPCB/OpenAQ expose.
- Attribution confidence is **hard-capped at "medium"** — it is a
  correlational heuristic by design and never claims causal certainty.
- The **monetized impact model stays Delhi-only** (see `docs/IMPACT_MODEL.md`
  §5) even though the forecast engine now covers all 3 cities — extending the
  health-impact arithmetic to Bengaluru/Indore needs their own city-specific
  mortality/morbidity assumptions, which we chose not to fabricate.

## 10. What Would Make This Real (Roadmap)

1. A pilot MoU with a city pollution-control board or health department to
   measure actual citizen action rates against a control period.
2. Swap observed-at-t weather for forecast weather in the feature pipeline.
3. Isolate the fire feature's real marginal contribution (e.g. an ablation
   run: same 3-city data, fire feature removed) now that enough Oct–Nov
   stubble-season data exists to test the original hypothesis properly.
4. Build Bengaluru/Indore-specific health-impact assumptions to extend the
   monetized impact model beyond Delhi now that both cities have trained,
   persistence-beating forecasts of their own.

## 11. Repository & Reproduction

- **GitHub:** https://github.com/ParvTiwari/vaayu-ai
- **Quick start:**
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  cp .env.example .env                          # API keys optional — system degrades gracefully
  python data/ingest/run_ingestion.py            # populate data/db/ (slow first time; cacheable)
  python models/train_forecast_model.py          # trains + writes models/BENCHMARKS.md
  uvicorn main:app --reload                      # API on :8000 (docs at /docs)
  streamlit run ui/app.py                         # UI on :8501
  pytest -q                                       # 55 tests
  ```
- A pre-built **data + model bundle** is included with this submission
  (`vaayu-ai-data-bundle.zip`) so the demo can be run immediately without
  waiting on live ingestion or API keys — see its included `README.txt` for
  drop-in instructions.

## 12. Repository Layout

```
data/ingest/       CPCB/OpenAQ, weather, FIRMS, Overpass ingestion  (+ README.md)
data/db/           parquet datasets + city_layers geojson + raw_cache  (gitignored)
models/            train_forecast_model.py · checkpoints/ · BENCHMARKS.md
agents/            orchestrator, forecast, attribution, enforcement, advisory, output, graph
main.py            FastAPI app (all routes)
ui/app.py          Streamlit demo frontend
tests/test_all.py  full pytest suite
docs/              this document · ARCHITECTURE.md · BENCHMARKS.md · IMPACT_MODEL.md · PITCH_SCRIPT.md
```

---

_All monetary figures use ₹83 ≈ US$1. Impact figures are a transparent
back-of-envelope model built on explicitly stated assumptions (§7,
[docs/IMPACT_MODEL.md](IMPACT_MODEL.md)), not a clinical or actuarial
projection. Model performance numbers (§6) are measured and reproducible via
`tests/test_all.py::test_forecast_reproduces_documented_metrics`._
