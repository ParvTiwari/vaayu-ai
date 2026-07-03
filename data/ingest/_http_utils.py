"""
Shared HTTP retry + raw-response caching helpers used by every module in
data/ingest/. Keeping this in one place avoids re-implementing backoff and
on-disk caching logic per source.

Three public getters share one retry/cache core (`_cached_request`):
  - cached_json_get   GET  -> parsed JSON   (CPCB, OpenAQ, Open-Meteo)
  - cached_text_get   GET  -> raw text      (NASA FIRMS returns CSV)
  - cached_post_json  POST -> parsed JSON   (Overpass wants a query body)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)


class CacheMiss(RuntimeError):
    """Raised on a cache miss when VAAYU_CACHE_ONLY is set (offline mode)."""

CACHE_DIR = Path(__file__).resolve().parent.parent / "db" / "raw_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_key_for(prefix: str, params: dict) -> str:
    """Build a stable, filesystem-safe cache key from a prefix + request params."""
    raw = prefix + "::" + json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}__{digest}"


def _cached_request(
    url: str,
    *,
    prefix: str,
    method: str,
    params: dict | None,
    data: dict | str | None,
    headers: dict | None,
    timeout: int,
    max_retries: int,
    cache_ext: str,
    parse: Callable[[str], Any],
) -> Any:
    """
    Core request loop shared by the public getters:
      - on-disk caching to data/db/raw_cache/ keyed by prefix + request shape,
        so repeated runs during development don't re-hit the API (important
        for slow, rate-limited sources like Overpass)
      - retries with exponential backoff (max_retries attempts) for network
        errors and retryable HTTP statuses (429, 5xx)
      - fail-fast (no retry) for auth errors (401/403) and other 4xx client
        errors, since retrying an identical bad request won't help

    Cached responses are stored as raw text and re-parsed on read, so a JSON
    getter always returns a fresh object (never a shared mutable across calls).
    Raises the underlying requests exception if all attempts are exhausted.
    """
    key = cache_key_for(prefix, {"url": url, "method": method, "params": params or {}, "data": data or ""})
    cache_file = CACHE_DIR / f"{key}.{cache_ext}"
    if cache_file.exists():
        logger.info(f"[cache hit] {prefix} -> {cache_file.name}")
        return parse(cache_file.read_text(encoding="utf-8"))

    # Offline / cache-only mode: never hit the network. Lets us rebuild a
    # dataset instantly from whatever a previous run already cached (e.g. when
    # an upstream API is slow/flaky). Callers that paginate treat this like a
    # normal fetch failure and stop cleanly.
    if os.getenv("VAAYU_CACHE_ONLY", "").strip():
        raise CacheMiss(f"cache-only mode: no cached response for {prefix}")

    last_exc: Exception | None = None
    retry_after: float | None = None  # seconds the server asked us to wait
    for attempt in range(1, max_retries + 1):
        retry_after = None
        try:
            resp = requests.request(
                method, url, params=params, data=data, headers=headers, timeout=timeout
            )
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
                # Rate-limited endpoints (Overpass) often tell us exactly how
                # long to back off — honour it rather than guessing.
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        retry_after = None
            else:
                # Any other 4xx raises immediately (not caught here, so it
                # propagates without wasting retries); 2xx returns the data.
                resp.raise_for_status()
                text = resp.text
                result = parse(text)  # parse first so we never cache a body we can't read
                cache_file.write_text(text, encoding="utf-8")
                return result

        if attempt < max_retries:
            wait = retry_after if retry_after is not None else 2**attempt
            wait = min(wait, 60)  # never sleep more than a minute between tries
            logger.warning(
                f"[{prefix}] attempt {attempt}/{max_retries} failed: {last_exc}. "
                f"Retrying in {wait:.0f}s..."
            )
            time.sleep(wait)

    logger.error(f"[{prefix}] all {max_retries} attempts failed: {last_exc}")
    raise last_exc


def cached_json_get(
    url: str,
    *,
    prefix: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    max_retries: int = 3,
) -> dict[str, Any]:
    """GET `url` and return parsed JSON (see `_cached_request` for cache/retry semantics)."""
    return _cached_request(
        url, prefix=prefix, method="GET", params=params, data=None, headers=headers,
        timeout=timeout, max_retries=max_retries, cache_ext="json", parse=json.loads,
    )


def cached_text_get(
    url: str,
    *,
    prefix: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    max_retries: int = 3,
) -> str:
    """GET `url` and return the raw response body as text (e.g. FIRMS CSV)."""
    return _cached_request(
        url, prefix=prefix, method="GET", params=params, data=None, headers=headers,
        timeout=timeout, max_retries=max_retries, cache_ext="txt", parse=lambda t: t,
    )


def cached_post_json(
    url: str,
    *,
    prefix: str,
    data: dict | str,
    headers: dict | None = None,
    timeout: int = 180,
    max_retries: int = 3,
) -> dict[str, Any]:
    """POST `data` to `url` and return parsed JSON (Overpass query interpreter)."""
    return _cached_request(
        url, prefix=prefix, method="POST", params=None, data=data, headers=headers,
        timeout=timeout, max_retries=max_retries, cache_ext="json", parse=json.loads,
    )


__all__ = ["cached_json_get", "cached_text_get", "cached_post_json", "cache_key_for", "CACHE_DIR", "CacheMiss"]
