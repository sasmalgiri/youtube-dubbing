"""
Content-hash caching for the dubbing pipeline.

DISABLED — user requested no reuse. Every job runs every step from scratch
and the work artifacts are deleted at the end of each job.

All get_* functions return None.
All put_* functions are no-ops.
The CACHE_DISABLED flag below is the single point of control —
flip it back to False to re-enable caching.
"""
from __future__ import annotations

# Single point of control for the entire cache layer.
# True  = no reads, no writes, every job is fresh.
# False = original behavior (SHA-keyed reuse across jobs).
CACHE_DISABLED = True

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, List, Optional

# ── Cache root ────────────────────────────────────────────────────────────────

_DEFAULT_CACHE = Path(__file__).resolve().parent / "cache"
CACHE_DIR: Path = Path(os.environ.get("VOICEDUB_CACHE", str(_DEFAULT_CACHE)))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Sub-directories per cache type
_ASR_DIR   = CACHE_DIR / "asr"
_TRANS_DIR = CACHE_DIR / "translation"
_TTS_DIR   = CACHE_DIR / "tts"

for _d in (_ASR_DIR, _TRANS_DIR, _TTS_DIR):
    _d.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file (streamed, memory-efficient)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(*parts: str) -> str:
    """Return hex SHA-256 of concatenated strings."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


# ── ASR cache ─────────────────────────────────────────────────────────────────

def get_asr(audio_path: Path, model: str, language: str) -> Optional[List[dict]]:
    """Return cached ASR segments, or None if not cached."""
    if CACHE_DISABLED:
        return None
    key = _sha256_file(audio_path) + "_" + _sha256_str(model, language)
    cache_file = _ASR_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            print(f"[Cache] ASR hit ({key[:12]}…)", flush=True)
            return data
        except Exception:
            cache_file.unlink(missing_ok=True)
    return None


def put_asr(audio_path: Path, model: str, language: str, segments: List[dict]) -> None:
    """Persist ASR segments to cache."""
    if CACHE_DISABLED:
        return
    key = _sha256_file(audio_path) + "_" + _sha256_str(model, language)
    cache_file = _ASR_DIR / f"{key}.json"
    try:
        cache_file.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
        print(f"[Cache] ASR stored ({key[:12]}…, {len(segments)} segments)", flush=True)
    except Exception as e:
        print(f"[Cache] ASR write error: {e}", flush=True)


# ── Translation cache ─────────────────────────────────────────────────────────

def get_translation(source_text: str, engine: str, target_lang: str) -> Optional[str]:
    """Return cached translation, or None if not cached."""
    if CACHE_DISABLED:
        return None
    key = _sha256_str(source_text, engine, target_lang)
    cache_file = _TRANS_DIR / f"{key}.txt"
    if cache_file.exists():
        try:
            text = cache_file.read_text(encoding="utf-8")
            print(f"[Cache] Translation hit ({key[:12]}…)", flush=True)
            return text
        except Exception:
            cache_file.unlink(missing_ok=True)
    return None


def put_translation(source_text: str, engine: str, target_lang: str, translated: str) -> None:
    """Persist translated text to cache."""
    if CACHE_DISABLED:
        return
    key = _sha256_str(source_text, engine, target_lang)
    cache_file = _TRANS_DIR / f"{key}.txt"
    try:
        cache_file.write_text(translated, encoding="utf-8")
        print(f"[Cache] Translation stored ({key[:12]}…)", flush=True)
    except Exception as e:
        print(f"[Cache] Translation write error: {e}", flush=True)


# ── TTS cache ─────────────────────────────────────────────────────────────────

def get_tts(text: str, voice: str, rate: str, engine: str) -> Optional[bytes]:
    """Return cached TTS audio bytes, or None if not cached."""
    if CACHE_DISABLED:
        return None
    key = _sha256_str(text, voice, rate, engine)
    cache_file = _TTS_DIR / f"{key}.bin"
    if cache_file.exists():
        try:
            data = cache_file.read_bytes()
            print(f"[Cache] TTS hit ({key[:12]}…)", flush=True)
            return data
        except Exception:
            cache_file.unlink(missing_ok=True)
    return None


def put_tts(text: str, voice: str, rate: str, engine: str, audio_bytes: bytes) -> None:
    """Persist TTS audio bytes to cache."""
    if CACHE_DISABLED:
        return
    key = _sha256_str(text, voice, rate, engine)
    cache_file = _TTS_DIR / f"{key}.bin"
    try:
        cache_file.write_bytes(audio_bytes)
        print(f"[Cache] TTS stored ({key[:12]}…, {len(audio_bytes)//1024}KB)", flush=True)
    except Exception as e:
        print(f"[Cache] TTS write error: {e}", flush=True)


# ── Cache management ──────────────────────────────────────────────────────────

def cache_stats() -> dict:
    """Return cache size statistics."""
    def _dir_stats(d: Path) -> dict:
        files = list(d.glob("*"))
        total_bytes = sum(f.stat().st_size for f in files if f.is_file())
        return {"count": len(files), "size_mb": round(total_bytes / 1024**2, 2)}

    return {
        "asr":         _dir_stats(_ASR_DIR),
        "translation": _dir_stats(_TRANS_DIR),
        "tts":         _dir_stats(_TTS_DIR),
        "cache_dir":   str(CACHE_DIR),
    }


def clear_cache(older_than_days: int = 0) -> dict:
    """
    Clear cache entries.
    - older_than_days=0  → clear everything
    - older_than_days=N  → only entries not accessed in N days
    Returns count of deleted files.
    """
    cutoff = time.time() - older_than_days * 86400 if older_than_days > 0 else None
    deleted = 0
    for d in (_ASR_DIR, _TRANS_DIR, _TTS_DIR):
        for f in d.glob("*"):
            if not f.is_file():
                continue
            if cutoff is None or f.stat().st_atime < cutoff:
                f.unlink(missing_ok=True)
                deleted += 1
    print(f"[Cache] Cleared {deleted} entries", flush=True)
    return {"deleted": deleted}
