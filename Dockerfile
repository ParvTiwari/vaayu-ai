# Vaayu AI — single-image deployment (Render / any Docker host).
# Runs the FastAPI backend (internal :8000) and the Streamlit UI (public, on the
# host-provided $PORT) in one container. The UI reaches the API via
# VAAYU_API_URL=127.0.0.1:8000.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    VAAYU_API_URL=http://127.0.0.1:8000 \
    HOME=/app

# libgomp1 is the OpenMP runtime XGBoost links against.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code + the ~14 MB runtime bundle (raw_cache is excluded via .dockerignore).
COPY . .

# Run as a non-root user; make the app dir + runtime-writable paths theirs.
RUN mkdir -p data/db/advisories .streamlit \
 && useradd -m -u 1000 user \
 && chown -R user:user /app
USER user

# The listen port is provided at runtime via $PORT (Render); no fixed EXPOSE.
CMD ["bash", "start.sh"]
