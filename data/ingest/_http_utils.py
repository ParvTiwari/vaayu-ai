"""
Shared HTTP retry + raw-response caching helpers used by every module in
data/ingest/. Keeping this in one place avoids re-implementing backoff and
on-disk caching logic per source.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "db" / "raw_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_key_for(prefix: str, params: dict) -> str:
    """Build a stable, filesystem-safe cache key from a prefix + request params."""
    raw = prefix + "::" + json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}__{digest}"


def cached_json_get(
    url: str,
    *,
    prefix: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    GET `url` and return parsed JSON, with:
      - on-disk caching to data/db/raw_cache/ keyed by prefix + params (so
        repeated runs during development don't re-hit the API)
      - retries with exponential backoff (max_retries attempts) for network
        errors and retryable HTTP statuses (429, 5xx)
      - fail-fast (no retry) for auth errors (401/403) and other 4xx client
        errors, since retrying an identical bad request won't help

    Raises the underlying requests exception if all attempts are exhausted.
    """
    key = cache_key_for(prefix, {"url": url, "params": params or {}})
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        logger.info(f"[cache hit] {prefix} -> {cache_file.name}")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
        else:
            if resp.status_code in (401, 403):
                # Auth error — retrying the same bad key won't help. Fail fast.
                resp.raise_for_status()
            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = requests.HTTPError(
                    f"retryable status {resp.status_code} from {url}", response=resp
                )
            else:
                # Any other 4xx raises immediately (not caught here, so it
                # propagates without wasting retries); 2xx returns the data.
                resp.raise_for_status()
                data = resp.json()
                cache_file.write_text(json.dumps(data), encoding="utf-8")
                return data

        if attempt < max_retries:
            wait = 2**attempt
            logger.warning(
                f"[{prefix}] attempt {attempt}/{max_retries} failed: {last_exc}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    logger.error(f"[{prefix}] all {max_retries} attempts failed: {last_exc}")
    raise last_exc


__all__ = ["cached_json_get", "cache_key_for", "CACHE_DIR"]
