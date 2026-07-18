# Deploying Vaayu AI to Render (free, Docker)

One container runs both the FastAPI backend (internal `:8000`) and the Streamlit
UI (public, on Render's `$PORT`). Judges get a single public URL. The ~14 MB of
parquet + GeoJSON + model checkpoints ship in the repo; the 1.2 GB ingestion
cache does not.

> **Free-tier note:** Render's free web service **sleeps after ~15 min idle**, so
> the first visit after a quiet period waits ~50 s while it wakes, then is fast.
> Fine for a demo; open the link once yourself before judging to pre-warm it.

## 1. Commit the code + runtime bundle to GitHub

Render builds from your GitHub repo, so the data/model files (normally
`.gitignore`d) must be committed. Force-add just the ~14 MB runtime subset — this
does **not** change `.gitignore`:

```bash
# from the repo root
git add Dockerfile start.sh .dockerignore render.yaml
git add -f data/db/unified_history.parquet \
           data/db/fire_daily.parquet \
           data/db/city_layers/*.geojson \
           models/checkpoints/*.json
git commit -m "Add Render Docker deployment + runtime data bundle"
git push origin HEAD
```

(All runtime files are < 6 MB, so no git-LFS is needed.)

## 2. Create the service on Render

1. Sign in at https://render.com (GitHub login is easiest — no card required).
2. **New → Blueprint**, then select this repository. Render reads `render.yaml`
   and configures the `vaayu-ai` Docker web service automatically.
   - *(Alternative:* **New → Web Service** → pick the repo → Render detects the
     `Dockerfile`; set Instance Type = **Free**.)*
3. Click **Apply / Create**. The first build takes a few minutes.
4. When it goes live, open the URL: `https://vaayu-ai.onrender.com`
   (Render may add a suffix if the name is taken).

## 3. (Optional) Enable live LLM localization

Without an LLM key the advisory agent uses the deterministic CPCB fallback —
fully functional, Hindi included. To turn on live LLM rephrasing:

- In the Render dashboard → your service → **Environment** → add
  `LLM_API_KEY` = your Groq / Anthropic / OpenAI-compatible key, then redeploy.

## Notes & troubleshooting

- **Updating the demo:** push to the same branch — Render auto-deploys.
- **Pre-warm before judging:** open the link once so the first judge doesn't hit
  the ~50 s cold start.
- **`runtime: docker` errors in the blueprint:** older Render accounts use
  `env: docker` instead — swap that one key in `render.yaml` if the blueprint is
  rejected.
- **Health check:** Render pings `/_stcore/health` (Streamlit's own endpoint);
  the API's `GET /docs` is the Swagger UI if you want to exercise routes directly.
- **Container writes:** generated advisory audio goes to `data/db/advisories/`
  (ephemeral on Render — fine; regenerated on demand).
