from __future__ import annotations

"""
Supabase metrics module for the YouTube dubbing app.

=== SQL SCHEMA — run this in your Supabase SQL editor once ===

-- Jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id                   TEXT PRIMARY KEY,
    url                  TEXT,
    source_language      TEXT,
    target_language      TEXT,
    tts_engine           TEXT,
    asr_model            TEXT,
    translation_engine   TEXT,
    started_at           TIMESTAMPTZ DEFAULT NOW(),
    completed_at         TIMESTAMPTZ,
    status               TEXT DEFAULT 'running',   -- running / done / error
    total_segments       INT,
    pass_rate_first_try  FLOAT,                    -- 0.0 – 1.0
    avg_duration_error_ms FLOAT,
    total_rerenders      INT,
    manual_review_count  INT,
    total_render_time_s  FLOAT,
    video_title          TEXT,
    output_url           TEXT
);

-- Job segments table
CREATE TABLE IF NOT EXISTS job_segments (
    id                  BIGSERIAL PRIMARY KEY,
    job_id              TEXT REFERENCES jobs(id) ON DELETE CASCADE,
    segment_idx         INT,
    start_time          FLOAT,
    end_time            FLOAT,
    source_text         TEXT,
    translated_text     TEXT,
    emotion             TEXT DEFAULT 'neutral',
    tts_engine          TEXT,
    tts_duration_s      FLOAT,
    expected_duration_s FLOAT,
    duration_error_ms   FLOAT,
    rerender_count      INT DEFAULT 0,
    qc_status           TEXT DEFAULT 'pass',       -- pass / manual_review / error
    qc_flags            JSONB DEFAULT '[]',
    manual_review       BOOL DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    -- upsert key: one row per (job_id, segment_idx)
    UNIQUE (job_id, segment_idx)
);

=== END SQL SCHEMA ===
"""

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50  # max rows per upsert request to Supabase


# ---------------------------------------------------------------------------
# No-op fallback (used when supabase-py is missing or env vars are absent)
# ---------------------------------------------------------------------------

class NoOpMetrics:
    """Silent fallback — every method is a no-op that never raises."""

    def record_job_start(self, job_id: str, url: str, settings: dict) -> None:
        pass

    def record_job_complete(self, job_id: str, status: str, metrics: dict) -> None:
        pass

    def record_segments(self, job_id: str, segments_data: list) -> None:
        pass

    def get_job_stats(self, job_id: str) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Real implementation
# ---------------------------------------------------------------------------

class SupabaseMetrics:
    """
    Fire-and-forget Supabase metrics client.

    All mutating methods spawn a daemon thread so they never block or crash
    the dubbing pipeline, even if Supabase is unreachable.
    """

    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        from supabase import create_client  # type: ignore[import]

        self._client = create_client(supabase_url, supabase_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_job_start(self, job_id: str, url: str, settings: dict) -> None:
        """Insert a new job row with status='running'."""
        payload: dict[str, Any] = {
            "id": job_id,
            "url": url,
            "status": "running",
            "source_language": settings.get("source_language"),
            "target_language": settings.get("target_language"),
            "tts_engine": settings.get("tts_engine"),
            "asr_model": settings.get("asr_model"),
            "translation_engine": settings.get("translation_engine"),
            "video_title": settings.get("video_title"),
        }
        self._fire(self._upsert_job, payload)

    def record_job_complete(self, job_id: str, status: str, metrics: dict) -> None:
        """
        Update an existing job row with final status and aggregate metrics.

        Expected keys in *metrics* (all optional):
            total_segments, pass_rate_first_try, avg_duration_error_ms,
            total_rerenders, manual_review_count, total_render_time_s,
            output_url, completed_at
        """
        payload: dict[str, Any] = {"id": job_id, "status": status}
        _copy_keys(
            metrics,
            payload,
            (
                "total_segments",
                "pass_rate_first_try",
                "avg_duration_error_ms",
                "total_rerenders",
                "manual_review_count",
                "total_render_time_s",
                "video_title",
                "output_url",
                "completed_at",
            ),
        )
        # Default completed_at to server time when caller omits it
        if "completed_at" not in payload:
            payload["completed_at"] = _utcnow_iso()
        self._fire(self._upsert_job, payload)

    def record_segments(self, job_id: str, segments_data: list) -> None:
        """
        Upsert segment rows in batches of 50.

        Each item in *segments_data* should be a dict with keys matching the
        job_segments columns. job_id is injected automatically if absent.
        """
        if not segments_data:
            return
        rows: list[dict[str, Any]] = []
        for seg in segments_data:
            row: dict[str, Any] = dict(seg)
            row["job_id"] = job_id
            rows.append(row)
        self._fire(self._upsert_segments_batched, rows)

    def get_job_stats(self, job_id: str) -> dict:
        """
        Fetch a single job row synchronously.

        Returns an empty dict on any failure so callers never need to handle
        exceptions.
        """
        try:
            response = (
                self._client.table("jobs")
                .select("*")
                .eq("id", job_id)
                .maybe_single()
                .execute()
            )
            data = response.data
            if data is None:
                return {}
            return dict(data)
        except Exception:
            logger.exception("SupabaseMetrics.get_job_stats failed for job_id=%s", job_id)
            return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire(self, fn, *args) -> None:
        """Run *fn* in a daemon thread — fire-and-forget."""
        t = threading.Thread(target=self._safe_call, args=(fn, *args), daemon=True)
        t.start()

    @staticmethod
    def _safe_call(fn, *args) -> None:
        try:
            fn(*args)
        except Exception:
            logger.exception("SupabaseMetrics background task failed: %s", fn.__name__)

    def _upsert_job(self, payload: dict) -> None:
        self._client.table("jobs").upsert(payload).execute()

    def _upsert_segments_batched(self, rows: list) -> None:
        for i in range(0, len(rows), _BATCH_SIZE):
            batch = rows[i : i + _BATCH_SIZE]
            self._client.table("job_segments").upsert(
                batch, on_conflict="job_id,segment_idx"
            ).execute()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _copy_keys(src: dict, dst: dict, keys: tuple) -> None:
    """Copy keys that exist in *src* into *dst*."""
    for k in keys:
        if k in src:
            dst[k] = src[k]


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def _build_metrics():
    """
    Attempt to build a real SupabaseMetrics instance.

    Falls back to NoOpMetrics if:
      - supabase-py is not installed
      - SUPABASE_URL or SUPABASE_KEY env vars are missing
      - any other initialisation error occurs
    """
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()

    if not url or not key:
        logger.debug(
            "SUPABASE_URL or SUPABASE_KEY not set — metrics disabled (NoOpMetrics)."
        )
        return NoOpMetrics()

    try:
        instance = SupabaseMetrics(url, key)
        logger.info("SupabaseMetrics initialised (url=%s…)", url[:40])
        return instance
    except ImportError:
        logger.warning(
            "supabase-py not installed — metrics disabled. "
            "Install with: pip install supabase"
        )
        return NoOpMetrics()
    except Exception:
        logger.exception("SupabaseMetrics init failed — falling back to NoOpMetrics.")
        return NoOpMetrics()


# Singleton — import this directly or call get_metrics()
metrics: SupabaseMetrics | NoOpMetrics = _build_metrics()


def get_metrics() -> SupabaseMetrics | NoOpMetrics:
    """Return the module-level metrics singleton."""
    return metrics
