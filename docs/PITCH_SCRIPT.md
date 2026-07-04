# Vaayu AI — 3-Minute Pitch Video Script

**Total spoken length:** ~420 words (~3:00 at a natural presenting pace).
Format below: **`[ON SCREEN]`** = what the viewer sees / the presenter does;
**NARRATION** = the exact words to record.

---

## 0:00 – 0:30 · The problem

`[ON SCREEN]` Title card "Vaayu AI", then the two stats appearing as text:
**1.67M deaths/year** · **900+ CPCB stations, only 31% actionable.**

> **NARRATION.** Every year, air pollution kills roughly 1.67 million people in
> India. To fight it, the government built a network of over 900 real-time air
> quality monitoring stations. But here's what a CAG audit found: only 31% of them
> have any actionable response protocol. The data exists — but the intelligence
> layer to act on it does not. That's the gap Vaayu AI fills.

## 0:30 – 1:15 · Live demo — forecast beats the baseline, then the map

`[ON SCREEN]` The BENCHMARKS table on screen; highlight the 24h row
(**91.29 vs 109.49 RMSE, +16.6%**). Then cut to the live Delhi map.

> **NARRATION.** Vaayu forecasts air quality 24, 48, and 72 hours ahead. The
> baseline everyone falls back on is persistence — just assume tomorrow looks like
> today. Our XGBoost model beats it at every horizon: at 24 hours we cut forecast
> error by 16.6%, at 48 hours by nearly 20%. That's an 18-point drop in AQI error —
> the difference between warning a city before a pollution spike and reacting after
> it. Here's Delhi, live. Each dot is a real monitoring station, colored by CPCB
> category. I'll turn on the forecast overlay — you can see where the air is headed.
> And this interpolated heatmap fills the gaps between stations, so every
> neighborhood gets a reading, not just the ones with a sensor.

## 1:15 – 2:00 · Enforcement priorities + citizen advisory

`[ON SCREEN]` Toggle the enforcement priority-zone layer; hover the top zone. Then
the chat box — type the query, show the English answer, switch language to Hindi,
play the audio clip.

> **NARRATION.** For regulators, Vaayu ranks enforcement priority. The formula is
> one sentence: risk times vulnerability — AQI severity multiplied by the hospitals,
> schools, and elderly-care homes within a kilometer. This zone tops the list:
> severe air, nineteen hospitals nearby. No black box — you can read exactly why.
> And for citizens, just ask. "Is it safe to jog near ITO tomorrow morning?"
> Vaayu routes the question, pulls the forecast, and answers — in English, or in
> Hindi. Every advisory cites the exact station and reading time behind it. And the
> health advice itself comes from a fixed CPCB table — the AI only translates it, it
> never invents a health claim.

## 2:00 – 2:40 · Architecture — the auditable guardrail

`[ON SCREEN]` 15-second hold on the ARCHITECTURE.md agent diagram; visually
highlight the deterministic agents vs. the single LLM box.

> **NARRATION.** Here's why you can trust it. Vaayu is six agents on a LangGraph
> pipeline — but look at the split. The forecast is machine learning. The source
> scoring, the enforcement ranking, and the health guidance are all deterministic
> rules: hardcoded, testable, auditable. The language model sits on the outside — it
> rephrases and it translates, but it never makes a decision. So every recommendation
> Vaayu hands a regulator or a citizen traces back to a formula and a real reading,
> not a black box. Fifty-five automated tests keep it honest.

## 2:40 – 3:00 · Impact + call to action

`[ON SCREEN]` The IMPACT_MODEL headline number; end on the Vaayu logo + one-line
scaling statement.

> **NARRATION.** The payoff: earlier, trustworthy warnings mean people act before the
> spike, not after. In Delhi alone we estimate hundreds of avoided hospital visits a
> year — on the order of ₹177 million in avoided costs over five years — and the same
> engine plugs into every one of India's 900-plus stations. The data is already
> there. Vaayu makes it act. Thank you.

---

### Delivery notes
- Pace ~140 words/min; the script is ~420 words, leaving a small buffer at each cut.
- Numbers to keep exact on screen: **+16.6% / +19.8% / +18.3%** (24/48/72h RMSE),
  **91.29 vs 109.49** (24h), **₹177M / 5 years**.
- If the LLM/TTS keys aren't configured for the recording, the Hindi advisory still
  renders from the deterministic CPCB table (no audio) — say "in Hindi" over the
  on-screen Devanagari text and skip the audio beat.
- Backup if the map is slow: pre-load the Delhi map before recording (layers cache on
  first load), or screen-record the map section separately and cut it in.
