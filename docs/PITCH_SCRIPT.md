# Vaayu AI — Pitch Video Script (target 3:30–3:45)

**Total spoken length:** ~525 words (~3:45 at a natural 140 wpm presenting
pace) — inside the hackathon's required 3–4 minute window, but with little
room left to slow down further; pick up the pace slightly on the middle beats
if you're running long, rather than the numbers themselves.
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

`[ON SCREEN]` The BENCHMARKS table on screen; highlight the Delhi 24h row
(**87.56 vs 109.49 RMSE, +20.0%**), then a quick glance down at the Bengaluru and
Indore rows. Then cut to the live Delhi map.

> **NARRATION.** Vaayu forecasts air quality 24, 48, and 72 hours ahead — and it's
> trained and running in all three pilot cities: Delhi, Bengaluru, and Indore. The
> baseline everyone falls back on is persistence — just assume tomorrow looks like
> today. Our XGBoost model beats it at every horizon, in every city: in Delhi we cut
> forecast error by 20% at 24 hours, and over 22% at 48. That's a 22-point drop in
> AQI error — the difference between warning a city before a pollution spike and
> reacting after it. Here's Delhi, live. Each dot is a real monitoring station,
> colored by CPCB category. I'll turn on the forecast overlay — you can see where
> the air is headed. And this interpolated heatmap fills the gaps between stations,
> so every neighborhood gets a reading, not just the ones with a sensor.

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

## 2:40 – 3:05 · Honest limitations — what we're not overclaiming

`[ON SCREEN]` A quick cut to the "Honest caveats" section of BENCHMARKS.md.

> **NARRATION.** And we're upfront about what's still uncertain. We added real
> satellite fire data for stubble-burning across all three cities — but its own
> contribution to accuracy is still modest, so we're not claiming it's the reason we
> beat the baseline until we can isolate that properly. We'd rather show you an
> honest, measured result than an inflated one.

## 3:05 – 3:30 · Impact + call to action

`[ON SCREEN]` The IMPACT_MODEL headline number; end on the Vaayu logo + one-line
scaling statement.

> **NARRATION.** The payoff: earlier, trustworthy warnings mean people act before the
> spike, not after. In Delhi alone we estimate hundreds of avoided hospital visits a
> year — on the order of ₹177 million in avoided costs over five years — and the same
> engine plugs into every one of India's 900-plus stations. The data is already
> there. Vaayu makes it act. Thank you.

---

### Delivery notes
- Pace ~140 words/min; the script is ~525 words — targets 3:45, close to the
  4:00 ceiling, so keep a brisk (not rushed) pace on the middle beats and only
  slow down for the numbers on screen. Don't rush below 3:00; the submission
  requires a 3–4 minute video.
- Numbers to keep exact on screen: **+20.0% / +22.5% / +20.8%** (Delhi 24/48/72h
  RMSE), **87.56 vs 109.49** (Delhi 24h), **₹177M / 5 years**. Bengaluru
  (+19.2–20.2%) and Indore (+17.3–19.7%) beat persistence too if you want a
  second on-screen proof point.
- If the LLM/TTS keys aren't configured for the recording, the Hindi advisory still
  renders from the deterministic CPCB table (no audio) — say "in Hindi" over the
  on-screen Devanagari text and skip the audio beat.
- Backup if the map is slow: pre-load the Delhi map before recording (layers cache on
  first load), or screen-record the map section separately and cut it in.
