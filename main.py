"""
Vaayu AI — FastAPI backend entrypoint.

Routes for /forecast, /attribution, /heatmap, /enforcement-priorities,
/advisory, and /query will be added here as each agent is implemented.
"""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env from the project root before any agent module reads os.getenv().
load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(
    title="Vaayu AI",
    description="Urban Air Quality Intelligence — ET AI Hackathon 2026, Problem Statement 5",
    version="0.1.0",
)


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "vaayu-ai"}
