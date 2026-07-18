#!/usr/bin/env bash
# Launch the FastAPI backend on an internal port, wait until it answers, then
# start the public Streamlit UI. Streamlit is the foreground process so the
# container's lifecycle tracks it.
set -euo pipefail

# Backend — internal only; the UI talks to it over localhost.
uvicorn main:app --host 127.0.0.1 --port 8000 &

# Wait (up to ~30s) for the API to become ready so the first UI request works.
python - <<'PY'
import time, urllib.request
for _ in range(30):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/", timeout=2)
        print("API ready")
        break
    except Exception:
        time.sleep(1)
else:
    print("API not ready after 30s; starting UI anyway")
PY

# Frontend — public. Render injects $PORT the service must bind to; default to
# 7860 for local `docker run`.
exec streamlit run ui/app.py \
  --server.port="${PORT:-7860}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
