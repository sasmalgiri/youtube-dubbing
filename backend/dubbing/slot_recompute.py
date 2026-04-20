"""Slot Recompute — rewrite segment timeline so audio and video actually match.

Three modes:
  "original"    — old behavior: trim/pad audio to fit original SRT slots (no-op here)
  "capped"      — speed audio up to max_audio_speedup, then expand slots by word-weight
  "audio_first" — never touch audio; slots = raw TTS duration, video adapts fully

Called AFTER TTS renders all segments but BEFORE final video assembly.
Each cue gets: tts_duration, audio_speedup, new_start, new_end, video_speed.
"""
from __future__ import annotations
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from .contracts import (
    Cue,
    MAX_AUDIO_SPEEDUP,
    MIN_VIDEO_SPEED,
    SLOT_DRIFT_WARN_MS,
    SLOT_DRIFT_FAIL_MS,
)

log = logging.getLogger(__name__)


# ─── Probe TTS duration ────────────────────────────────────────────────

def probe_duration(wav_path: Path, ffprobe: str = "ffprobe") -> float:
    """Get exact audio duration in seconds using ffprobe."""
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {wav_path}: {result.stderr.strip()}")
    raw = result.stdout.strip()
    if not raw:
        raise RuntimeError(f"ffprobe returned empty output for {wav_path}")
    try:
        dur = float(raw)
    except ValueError:
        raise RuntimeError(f"ffprobe returned non-numeric output for {wav_path}: {raw!r}")
    if dur <= 0:
        raise RuntimeError(f"ffprobe returned non-positive duration for {wav_path}: {dur}")
    return dur


def measure_tts_durations(
    cues: List[Cue],
    tts_wav_dir: Path,
    wav_pattern: str = "seg_{id:04d}.wav",
    ffprobe: str = "ffprobe",
) -> List[Cue]:
    """Populate cue.tts_duration from rendered TTS wav files.

    wav_pattern: format string with {id} placeholder matching cue.id.
    """
    for cue in cues:
        wav_name = wav_pattern.format(id=cue.id)
        wav_path = tts_wav_dir / wav_name
        if wav_path.exists():
            try:
                cue.tts_duration = probe_duration(wav_path, ffprobe)
            except Exception as e:
                log.warning("Probe failed for cue %d (%s): %s — using original slot",
                            cue.id, wav_path, e)
                cue.tts_duration = cue.duration
        else:
            log.warning("TTS wav missing for cue %d: %s", cue.id, wav_path)
            cue.tts_duration = cue.duration  # fallback to original slot
    return cues


# ─── Core: recompute slots ─────────────────────────────────────────────

def recompute_slots(
    cues: List[Cue],
    mode: str = "capped",
    max_speedup: float = MAX_AUDIO_SPEEDUP,
    min_video_speed: float = MIN_VIDEO_SPEED,
    drift_warn_ms: float = SLOT_DRIFT_WARN_MS,
    drift_fail_ms: float = SLOT_DRIFT_FAIL_MS,
) -> List[Cue]:
    """Rewrite cue.new_start / new_end / video_speed based on TTS durations.

    Requires cue.tts_duration to be set (call measure_tts_durations first).

    Returns the same cue list with recomputed timing fields.
    """
    if mode == "original":
        # No-op: keep original slots, audio gets trimmed/padded elsewhere
        for cue in cues:
            cue.new_start = cue.start
            cue.new_end = cue.end
            cue.video_speed = 1.0
            cue.audio_speedup = 1.0
            cue.slot_drift_ms = 0.0
            cue.slot_status = "ok"
        return cues

    if not cues:
        return cues

    # Sort cues by start time — required for gap preservation in STEP 4
    cues.sort(key=lambda c: c.start)

    # Clean up stale state from any prior call (idempotency)
    _OWNED_FLAGS = frozenset({
        "zero_duration_cue_expanded", "video_speed_below_floor",
    })
    for cue in cues:
        if hasattr(cue, '_new_slot_dur'):
            del cue._new_slot_dur
        cue.qc_flags = [f for f in cue.qc_flags
                        if not any(f.startswith(prefix) for prefix in _OWNED_FLAGS)]

    # Ensure all cues have TTS durations
    for cue in cues:
        if cue.tts_duration is None:
            log.warning("Cue %d missing tts_duration, using original slot", cue.id)
            cue.tts_duration = cue.duration

    # ── STEP 1: Per-segment audio speedup (capped mode only) ───────────
    for cue in cues:
        S = cue.duration        # original slot
        # Recover raw TTS duration if this is a re-run (idempotency)
        D = cue.tts_duration * cue.audio_speedup if cue.audio_speedup > 1.0 else cue.tts_duration
        cue.audio_speedup = 1.0  # reset before recomputing
        cue.tts_duration = D     # restore raw TTS duration for all branches

        if S <= 0:
            # Zero-duration cue (bad ASR) — skip speedup
            continue

        if mode == "capped" and D > S:
            speed = min(D / S, max_speedup)
            cue.audio_speedup = round(speed, 4)
            cue.tts_duration = round(D / speed, 4)
        elif mode == "audio_first":
            # Never speed up audio
            cue.audio_speedup = 1.0
        else:
            # TTS fits in slot already
            cue.audio_speedup = 1.0

    # ── STEP 2: Measure total overflow ─────────────────────────────────
    total_original = sum(c.duration for c in cues)
    total_audio = sum(c.tts_duration for c in cues)
    overflow = total_audio - total_original

    if overflow <= 0.01:
        # Everything fits — keep original timeline, just record speedups
        for cue in cues:
            cue.new_start = cue.start
            cue.new_end = cue.end
            cue.video_speed = 1.0
            cue.slot_drift_ms = round((cue.duration - cue.tts_duration) * 1000, 1)
            cue.slot_status = _classify_drift(cue.slot_drift_ms, drift_warn_ms, drift_fail_ms)
        return cues

    # ── STEP 3: Distribute overflow by word-weight ─────────────────────
    total_words = sum(c.word_count for c in cues)
    if total_words == 0:
        total_words = len(cues)  # safety: equal weight

    for cue in cues:
        w = cue.word_count or 1  # fallback: equal weight per cue
        weight = w / total_words
        extra = overflow * weight
        new_slot = cue.duration + extra
        # Ensure new slot is at least as long as the TTS audio
        new_slot = max(new_slot, cue.tts_duration)
        cue._new_slot_dur = new_slot

    # Re-normalize: max() clamp may have inflated total beyond target.
    # Iteratively shrink non-clamped cues until excess is absorbed.
    target_total = total_audio  # we want total timeline == total audio
    for _pass in range(5):  # max 5 passes to converge
        actual_total = sum(c._new_slot_dur for c in cues)
        excess = actual_total - target_total
        if excess <= 0.001:
            break  # converged
        shrinkable = [c for c in cues if c._new_slot_dur > c.tts_duration + 0.001]
        if not shrinkable:
            # All cues are at their TTS floor — cannot shrink further.
            # Timeline will be longer than target. Log and accept.
            log.warning(
                "Slot recompute: all cues at TTS floor, cannot absorb %.2fs excess. "
                "Timeline will be %.2fs longer than target.",
                excess, excess,
            )
            break
        shrink_total = sum(c._new_slot_dur - c.tts_duration for c in shrinkable)
        for cue in shrinkable:
            slack = cue._new_slot_dur - cue.tts_duration
            reduction = min(excess * (slack / shrink_total), slack)
            cue._new_slot_dur -= reduction
            cue._new_slot_dur = max(cue._new_slot_dur, cue.tts_duration)

    # ── STEP 4: Build new timeline preserving inter-cue gaps ───────────
    cursor = cues[0].start  # preserve original start offset
    for i, cue in enumerate(cues):
        if i > 0:
            original_gap = cue.start - cues[i - 1].end
            cursor += max(original_gap, 0)  # preserve silence gaps
        cue.new_start = round(cursor, 4)
        cue.new_end = round(cursor + cue._new_slot_dur, 4)
        cursor = cue.new_end

    # ── STEP 5: Per-segment video speed ────────────────────────────────
    for cue in cues:
        original_dur = cue.duration
        new_dur = cue.new_end - cue.new_start
        if new_dur > 0 and original_dur > 0:
            cue.video_speed = round(original_dur / new_dur, 4)
        else:
            cue.video_speed = 1.0
            if original_dur <= 0 and new_dur > 0:
                cue.qc_flags.append("zero_duration_cue_expanded")
                log.warning("Cue %d has zero original duration but expanded slot %.2fs",
                            cue.id, new_dur)

        # Floor check — clamp to min_video_speed to prevent assembly failures
        if cue.video_speed < min_video_speed:
            cue.qc_flags.append(
                f"video_speed_below_floor:{cue.video_speed:.2f}<{min_video_speed}"
            )
            log.warning(
                "Cue %d video speed %.2fx below floor %.2fx — clamped to floor",
                cue.id, cue.video_speed, min_video_speed,
            )
            cue.video_speed = min_video_speed

    # ── STEP 6: Compute drift ──────────────────────────────────────────
    for cue in cues:
        new_dur = cue.new_end - cue.new_start
        cue.slot_drift_ms = round((new_dur - cue.tts_duration) * 1000, 1)
        cue.slot_status = _classify_drift(cue.slot_drift_ms, drift_warn_ms, drift_fail_ms)

    # Cleanup temp attr
    for cue in cues:
        if hasattr(cue, '_new_slot_dur'):
            del cue._new_slot_dur

    return cues


# ─── Helpers ────────────────────────────────────────────────────────────

def _classify_drift(drift_ms: float, warn: float, fail: float) -> str:
    """Classify drift into ok/warn/fail."""
    if warn >= fail:
        warn, fail = fail, warn  # ensure warn < fail
    if abs(drift_ms) < warn:
        return "ok"
    elif abs(drift_ms) < fail:
        return "warn"
    return "fail"


def summary(cues: List[Cue]) -> dict:
    """Return a summary dict of the recompute results."""
    if not cues:
        return {}

    total_original = sum(c.duration for c in cues)
    total_new = sum(c.new_duration for c in cues)
    ok = sum(1 for c in cues if c.slot_status == "ok")
    warn = sum(1 for c in cues if c.slot_status == "warn")
    fail = sum(1 for c in cues if c.slot_status == "fail")
    speeds = [c.video_speed for c in cues if c.video_speed != 1.0]

    return {
        "total_cues": len(cues),
        "total_original_s": round(total_original, 2),
        "total_new_s": round(total_new, 2),
        "expansion_pct": round((total_new - total_original) / total_original * 100, 1)
            if total_original > 0 else 0.0,
        "ok": ok,
        "warn": warn,
        "fail": fail,
        "video_speed_min": round(min(speeds), 3) if speeds else 1.0,
        "video_speed_max": round(max(speeds), 3) if speeds else 1.0,
    }
