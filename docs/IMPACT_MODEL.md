# Vaayu AI — Impact Model

> **Read this as a transparent back-of-envelope, not a clinical estimate.** Every
> number is a stated assumption or a measured result. The point is to show the
> *shape* of the impact and — critically — **which assumptions the result is most
> sensitive to** (§4), so a reviewer can pressure-test it rather than take it on faith.

## 1. The problem (from the ET AI Hackathon 2026 PS5 brief)

- **~1.67 million premature deaths/year** in India from air pollution.
- **900+ CAAQMS monitoring stations**, but a CAG audit found **only 31% have
  actionable response protocols** — i.e. ~69% of stations produce numbers that
  nobody converts into a decision.

Vaayu AI is precisely that missing **"actionable response" layer**: it reads the same
station feeds and produces forecasts (act *before* the spike), source attribution
(what to act *on*), enforcement prioritization (where regulators act *first*), and
citizen advisories (what individuals *do*). The core intervention it enables is
**advance, trustworthy warning** — turning a same-day reading into a 24–72h heads-up.

## 2. What we measured (the engine of the impact)

| Horizon | Persistence RMSE | Vaayu model RMSE | Improvement |
|---|---:|---:|:--:|
| 24h | 109.49 | 91.29 | **+16.6%** |
| 48h | 118.28 | 94.92 | **+19.8%** |
| 72h | 118.95 | 97.15 | **+18.3%** |

_(AQI points, Delhi held-out test set; see `docs/BENCHMARKS.md`.)_ At 24h the model
cuts mean forecast error by ~18 AQI points versus the naive "tomorrow = today"
baseline. **This accuracy is the lever for the entire chain below:** persistence fails
hardest exactly on *transition days* (when a pollution episode begins or clears) —
the days advance action matters most — and people only act on alerts they can trust.

## 3. Back-of-envelope health impact (Delhi pilot)

We deliberately model **acute morbidity** (ER visits / hospital admissions from
short-term exposure), not chronic mortality, because short-term warnings act on
short-term exposure. The chain is a product of independent factors:

```
avoided acute events  =  E_high  ×  reach  ×  act  ×  eff
```

**Every factor is a stated assumption:**

| Sym | Assumption | Value | Basis / note |
|---|---|---|---|
| A1 | Delhi NCT population | 20,000,000 | 2025 estimate |
| A2 | Delhi premature deaths/yr attributable to air pollution | 12,000 | **Conservative** end of published range (studies span 12,000–54,000) |
| A3 | Acute pollution-attributable hospital events : deaths | 20 : 1 | Morbidity ≫ mortality; rule-of-thumb → 240,000 acute events/yr |
| A4 | Share of acute events driven by short-term high-AQI episodes (warnable) | 40% | Delhi exposure concentrated in winter high-AQI days → **E_high = 96,000/yr** |
| A5 | Year-1 service reach (population receiving advisories) | 5% → 1,000,000 | Conservative for a new civic service |
| A6 | Reached users who take protective action given a reliable 24–72h warning | 25% | Behaviour-change literature; **enabled by forecast accuracy (§2)** |
| A7 | Reduction in an acting person's acute-exposure risk that day (mask / indoor / reschedule) | 30% | Avoidance efficacy |
| A8 | Direct cost per acute respiratory/cardiac hospital event | ₹30,000 | Indian cost studies (~₹20k–40k) |

**Year-1 result (Delhi):**

```
E_high  ×  reach  ×  act  ×  eff
96,000  ×  0.05   ×  0.25 ×  0.30   =  360 acute events avoided / year
360 events  ×  ₹30,000               =  ₹10.8 million / year  (~$130,000)
```

360 avoided hospitalizations/ER visits in year one, at a deliberately conservative 5%
adoption. The value is **linear in adoption** — the number scales directly as the
service reaches more people.

## 4. Sensitivity — which assumptions would halve (or multiply) the numbers

Because the headline is a **product of independent multipliers**, halving any single
factor halves the result. This is the honest, cheap disclosure a reviewer should
demand:

| If this assumption is wrong… | Base | Halved → | Effect on avoided events |
|---|---|---|---|
| **A5 reach** (adoption 5% → 2.5%) | 360 | **180** | ×0.5 |
| **A6 action rate** (25% → 12.5%) | 360 | **180** | ×0.5 — this is what a *less accurate* forecast erodes |
| **A7 avoidance efficacy** (30% → 15%) | 360 | **180** | ×0.5 |
| **A4 warnable share** (40% → 20%) | 360 | **180** | ×0.5 (via E_high) |
| **A3 morbidity ratio** (20:1 → 10:1) | 360 | **180** | ×0.5 (via E_high) |

And the upside band, using the **published high-end** death figure instead of our
conservative one:

| Alternative | Value | Year-1 avoided events | Year-1 ₹ saved |
|---|---|---:|---:|
| **Pessimistic** (any one key factor halved) | — | ~180 | ~₹5.4M |
| **Base case** (assumptions above) | A2 = 12,000 deaths | **360** | **₹10.8M** |
| **Optimistic** (published high-end mortality) | A2 = 54,000 deaths | ~1,620 | ~₹48.6M |

So the honest Year-1 Delhi range is **~180–1,620 avoided acute events** (base 360).
The two factors most worth scrutinizing are **A6 (action rate)** — because it depends
on forecast trust, which is exactly what our +16.6–19.8% accuracy edge buys — and
**A2/A3 (the mortality→morbidity chain)**, where the published mortality range alone
spans 4.5×.

## 5. Five-year scaling (Delhi, base-case assumptions)

Holding A2–A4, A7, A8 fixed and growing **adoption (A5)** as the service matures.
Delhi avoided events = `E_high × adoption × act × eff = 96,000 × adoption × 0.25 × 0.30`
(≈ 7,200 avoided events at 100% adoption, scaled linearly by the adoption fraction).

| Year | Adoption (A5) | People reached | Avoided acute events | ₹ saved | ~USD |
|---|---:|---:|---:|---:|---:|
| 1 | 5% | 1.0 M | 360 | ₹10.8 M | $130k |
| 2 | 10% | 2.0 M | 720 | ₹21.6 M | $260k |
| 3 | 15% | 3.0 M | 1,080 | ₹32.4 M | $390k |
| 4 | 22% | 4.4 M | 1,584 | ₹47.5 M | $572k |
| 5 | 30% | 6.0 M | 2,160 | ₹64.8 M | $781k |
| **5-yr cumulative (Delhi)** | | | **5,904 events** | **₹177 M** | **~$2.1 M** |

**Multi-city expansion (upside, not in the headline):** Bengaluru (~13 M) and Indore
(~3.5 M) have cleaner baseline air and far fewer high-AQI days, so their per-capita
`E_high` is much lower — we do **not** claim Delhi-equivalent numbers for them. As a
rough, separately-flagged estimate, at Year-5 adoption they might add on the order of
**+20–30%** to the Delhi figure (their OSM attribution/enforcement layers already run
today; only their AQI backfill is pending). We keep the headline Delhi-only because
that is the city we have measured.

## 6. Beyond the health arithmetic (harder to monetize, still real)

- **Regulatory targeting.** Enforcement prioritization (`risk × vulnerability`) tells a
  pollution-control board which of the 69%-without-a-protocol zones to inspect *first* —
  directing scarce inspection capacity at the highest-harm areas.
- **Source-specific action.** Attribution (traffic vs. industrial vs. stubble) supports
  the *right* intervention (traffic management vs. industrial audit vs. stubble
  coordination) instead of blanket measures.
- **Equity.** Vulnerability weighting explicitly prioritizes zones near hospitals,
  schools, and elderly-care facilities — protecting those least able to avoid exposure.

## 7. What would make these numbers real (not just modeled)

1. A pilot MoU with one city's pollution-control board or health department to measure
   *actual* action rates (A6) and avoided admissions against a control period.
2. Ingest Bengaluru/Indore AQI history to extend forecasting beyond Delhi.
3. Add the FIRMS fire feature (currently 0) — expected to sharpen Oct–Nov stubble-season
   forecasts, the highest-exposure window.
4. Swap observed-at-t weather for forecast weather to remove the mild optimism noted in
   `docs/BENCHMARKS.md`.

---

_All monetary figures use ₹83 ≈ US$1. Health figures are illustrative back-of-envelope
estimates built on the explicitly stated assumptions above, not clinical or actuarial
projections. The model performance numbers (§2) are measured and reproducible
(`docs/BENCHMARKS.md`, `tests/test_all.py`)._
