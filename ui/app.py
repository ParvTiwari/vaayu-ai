"""
Vaayu AI — Streamlit demo frontend.

Talks to the FastAPI backend (see main.py) over HTTP. Start the backend first:

    uvicorn main:app --reload            # terminal 1  (http://localhost:8000)
    streamlit run ui/app.py              # terminal 2  (http://localhost:8501)

Set VAAYU_API_URL to point at a non-default backend.

Sections:
  1. City selector + Folium map (station AQI markers, heatmap toggle, priority-zone
     toggle). Click a marker → station detail + "Get 24h forecast" button.
  2. Free-text "Ask about air quality" box → /query (intent inferred) with a
     language dropdown and an audio player if TTS was generated.
  3. "Model & Data Transparency" expander — persistence-baseline numbers, data
     sources, and the scoring formulas in plain text.
"""
from __future__ import annotations

import os
from pathlib import Path

import folium
import requests
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

API = os.getenv("VAAYU_API_URL", "http://localhost:8000").rstrip("/")
ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_PATH = ROOT / "models" / "BENCHMARKS.md"

# CPCB category → colour (Good→Severe).
CAT_COLOR = {
    "good": "#009865", "satisfactory": "#a3c853", "moderate": "#f6c700",
    "poor": "#f29305", "very_poor": "#e93f33", "severe": "#8f3f97",
}

st.set_page_config(page_title="Vaayu AI", layout="wide")


# --- backend calls (cached) --------------------------------------------------


def _get(path, **params):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=30)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            return {"_error": f"{r.status_code}: {str(detail)[:200]}"}
        return r.json()
    except requests.RequestException as exc:
        return {"_error": f"backend unreachable: {exc}"}


def _post(path, payload):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=90)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            return {"_error": f"{r.status_code}: {str(detail)[:200]}"}
        return r.json()
    except requests.RequestException as exc:
        return {"_error": f"backend unreachable: {exc}"}


@st.cache_data(ttl=300, show_spinner=False)
def get_stations(city):
    return _get(f"/stations/{city}")


@st.cache_data(ttl=300, show_spinner=False)
def get_enforcement(city):
    return _get(f"/enforcement-priorities/{city}")


@st.cache_data(ttl=300, show_spinner=False)
def get_heatmap(city):
    return _get(f"/heatmap/{city}")


@st.cache_data(ttl=300, show_spinner=False)
def get_forecast(city, lat, lon, station_id):
    return _get(f"/forecast/{city}", lat=lat, lon=lon, station_id=station_id)


def backend_alive():
    try:
        return requests.get(f"{API}/", timeout=5).status_code == 200
    except requests.RequestException:
        return False


# Pilot cities (match the backend config).
CITIES = ["Delhi", "Bengaluru", "Indore"]
CITY_CENTRE = {"Delhi": (28.61, 77.21), "Bengaluru": (12.97, 77.59), "Indore": (22.72, 75.86)}


# --- header ------------------------------------------------------------------

st.title("🌫️ Vaayu AI — Urban Air-Quality Intelligence")
if not backend_alive():
    st.error(
        f"Backend not reachable at {API}. Start it with "
        "`uvicorn main:app --reload` from the project root."
    )
    st.stop()

with st.sidebar:
    st.header("Controls")
    city = st.selectbox("City", CITIES, index=0)
    lang = st.selectbox(
        "Advisory language",
        [("English", "en"), ("हिन्दी (Hindi)", "hi"), ("ಕನ್ನಡ (Kannada)", "kn")],
        format_func=lambda x: x[0],
    )[1]
    st.divider()
    show_heatmap = st.toggle("Overlay interpolated AQI heatmap", value=False)
    show_zones = st.toggle("Overlay enforcement priority zones", value=False)
    st.caption("Layers are cached at first load for a fast demo.")


# --- Section 1: map ----------------------------------------------------------

st.subheader(f"1 · Live station map — {city}")

snap = get_stations(city)
stations = snap.get("stations", []) if "_error" not in snap else []
if "_error" in snap:
    st.warning(f"Stations: {snap['_error']}")
elif not stations:
    st.info(f"No ingested AQI stations for {city} yet — run the AQI backfill for this city. "
            "Attribution and enforcement layers still work from OSM/FIRMS data.")

if stations:
    clat = sum(s["lat"] for s in stations) / len(stations)
    clon = sum(s["lon"] for s in stations) / len(stations)
else:
    clat, clon = CITY_CENTRE.get(city, (22.0, 78.0))

fmap = folium.Map(location=[clat, clon], zoom_start=11, tiles="cartodbpositron")

if show_heatmap:
    hm = get_heatmap(city)
    if "_error" in hm:
        st.warning(f"Heatmap: {hm['_error']}")
    else:
        pts = [[f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0],
                min(f["properties"]["aqi"] / 500.0, 1.0)] for f in hm.get("features", [])]
        if pts:
            HeatMap(pts, radius=14, blur=18, min_opacity=0.35).add_to(fmap)

if show_zones:
    enf = get_enforcement(city)
    if "_error" in enf:
        st.warning(f"Priority zones: {enf['_error']}")
    else:
        for z in enf.get("priority_zones", []):
            folium.CircleMarker(
                location=[z["lat"], z["lon"]],
                radius=6 + z["priority_score"] / 8.0,
                color="#7b1fa2", weight=1, fill=True, fill_color="#7b1fa2",
                fill_opacity=0.25,
                tooltip=f"Priority {z['priority_score']} · {z['station_id']}",
            ).add_to(fmap)

# Station markers coloured by CPCB category (drawn on top).
for s in stations:
    color = CAT_COLOR.get(s["category"], "#888888")
    popup = folium.Popup(
        f"<b>{s['station_name']}</b><br>AQI {s['aqi']} — {(s['category'] or 'n/a').replace('_', ' ')}"
        f"<br><small>{s['timestamp'][:16]}</small>", max_width=260)
    folium.CircleMarker(
        location=[s["lat"], s["lon"]], radius=8, color="#222", weight=1,
        fill=True, fill_color=color, fill_opacity=0.9, popup=popup,
        tooltip=f"{s['station_name']} · AQI {s['aqi']}",
    ).add_to(fmap)

legend = " ".join(
    f"<span style='background:{c};padding:2px 6px;border-radius:3px;color:#fff'>{k.replace('_', ' ')}</span>"
    for k, c in CAT_COLOR.items())
st.markdown(f"<div style='font-size:12px'>{legend}</div>", unsafe_allow_html=True)

map_state = st_folium(fmap, height=520, width=None, returned_objects=["last_object_clicked"])

# Resolve the clicked marker to the nearest station.
clicked = (map_state or {}).get("last_object_clicked")
if clicked and stations:
    cl, co = clicked["lat"], clicked["lng"]
    sel = min(stations, key=lambda s: (s["lat"] - cl) ** 2 + (s["lon"] - co) ** 2)
    st.session_state["selected_station"] = sel

sel = st.session_state.get("selected_station")
if sel and sel.get("station_id") in {s.get("station_id") for s in stations}:
    st.markdown(f"**Selected:** {sel['station_name']} — current AQI **{sel['aqi']}** "
                f"({(sel['category'] or 'n/a').replace('_', ' ')})")
    if st.button(f"📈 Get 24/48/72h forecast for {sel['station_name']}"):
        fc = get_forecast(city, sel["lat"], sel["lon"], sel["station_id"])
        if "_error" in fc:
            st.warning(f"Forecast: {fc['_error']}")
        else:
            f = fc["forecast"]
            cols = st.columns(3)
            for col, (hk, hv) in zip(cols, f.get("horizons", {}).items()):
                col.metric(f"AQI +{hk}", hv["aqi"], help=(hv["category"] or "").replace("_", " "))
            st.caption(f"Model: {f['model']} · reference AQI {f.get('reference_aqi')} "
                       f"at {str(f.get('reference_timestamp'))[:16]}")
else:
    st.info("Click a station marker on the map to select it, then request a forecast.")


# --- Section 2: chat ---------------------------------------------------------

st.subheader("2 · Ask about air quality")
q = st.text_input("Free text", placeholder=f"Is it safe to jog near a station in {city} tomorrow morning?")
if st.button("Ask", type="primary") and q.strip():
    lat0, lon0 = CITY_CENTRE.get(city, (22.0, 78.0))
    payload = {"city": city, "lang": lang, "query": q, "lat": lat0, "lon": lon0}
    if sel:
        payload["lat"], payload["lon"], payload["station_id"] = sel["lat"], sel["lon"], sel["station_id"]
    with st.spinner("Running the multi-agent pipeline…"):
        res = _post("/query", payload)
    if "_error" in res:
        st.warning(f"Query: {res['_error']}")
    else:
        st.caption(f"Detected intent: **{res.get('query_type')}** · components run: "
                   f"{', '.join(res.get('components_run', [])) or 'none'}")
        advisory = (res.get("chat") or {}).get("advisory")
        if advisory:
            st.success(advisory)
        st.markdown((res.get("chat") or {}).get("summary", "_no summary_"))
        src = res.get("sources_cited")
        if src:
            st.caption(f"Source: {src.get('station_name')} · AQI {src.get('aqi')} "
                       f"({src.get('reading_source')}) · {str(src.get('reading_timestamp'))[:16]}")
        vp = res.get("voice_output_path")
        if vp and Path(vp).exists():
            st.audio(vp)


# --- Section 3: transparency -------------------------------------------------


def _benchmarks_results() -> str:
    if not BENCHMARKS_PATH.exists():
        return "_BENCHMARKS.md not found — run models/train_forecast_model.py._"
    text = BENCHMARKS_PATH.read_text(encoding="utf-8")
    out, capture = [], False
    for line in text.splitlines():
        if line.startswith("## Results"):
            capture = True
            out.append(line)
            continue
        if capture and line.startswith("## ") and "Verdict" not in line and "Results" not in line:
            break
        if capture:
            out.append(line)
    return "\n".join(out).strip() or "_(results section not found)_"


with st.expander("🔍 Model & Data Transparency (auditability)"):
    st.markdown("#### Forecast vs. persistence baseline")
    st.markdown(_benchmarks_results())

    st.markdown("#### Data sources")
    st.markdown(
        "- **AQI**: CPCB (`data.gov.in`) live + **OpenAQ v3** historical (approx. AQI from PM2.5/PM10 CPCB sub-indices)\n"
        "- **Weather**: Open-Meteo archive (temp, humidity, wind, precip, boundary-layer height)\n"
        "- **Active fires**: NASA FIRMS (VIIRS) — daily count within 100 km of each station\n"
        "- **Geospatial**: OpenStreetMap via Overpass — road-density grid, industrial land-use, vulnerability POIs"
    )

    st.markdown("#### Scoring formulas (deterministic, no LLM)")
    st.markdown(
        "**Attribution** (each score 0–1):\n"
        "- `traffic` = mean OSM road density within 2 km ÷ city 95th-percentile density\n"
        "- `industrial` = 1.0 inside a `landuse=industrial` polygon, else linear falloff to 0 at 3 km\n"
        "- `fire` = normalised FIRMS `nearby_fire_count`, forced 0 outside the Oct–Nov stubble season\n"
        "- `overall` = the highest of the three; **confidence capped at 'medium'** (correlational, not causal)\n\n"
        "**Enforcement** priority per zone:\n"
        "- `risk` = CPCB AQI category severity 1–6\n"
        "- `vulnerability` = weighted nearby POIs within 1 km (hospital/elderly-care = 3, school = 2) + generic road-density proxy\n"
        "- `priority_score` = 100 × risk × vulnerability ÷ worst-zone value (sorted descending)\n\n"
        "**Advisory** health facts come from a hardcoded CPCB guidance table; the LLM only rephrases/localises it "
        "and is instructed never to invent AQI numbers or health claims."
    )
