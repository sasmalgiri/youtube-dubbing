"""Voice-cloning pre-flight + reference-audio extraction.

Called by _tts_coqui_xtts (and any other cloning engine that opts in) before
model loading. Does three things:

  1. Verifies XTTS prerequisites (torch, CUDA, Coqui TTS package, language).
     Returns a structured diagnosis so the pipeline can fail fast with a
     clear message instead of mid-inference with a stack trace.

  2. Picks a *usable* reference audio clip — not just the first 15s of the
     source, which is almost always an intro/music/logo and poisons the
     voice clone. Prefers: user-provided WAVs → clean speech region from
     source audio → fallback to the naive first-N-seconds slice.

  3. Validates reference clip has non-trivial RMS energy so we don't feed
     XTTS a silent file and get garbage output.

All functions are pure-ish and don't touch global state. The pipeline passes
in its ffmpeg path and a progress callback.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# XTTS v2 supports these target languages for cloning. Both the raw cfg
# value (e.g. "hi", "zh") and the XTTS-specific code (e.g. "zh-cn") are
# accepted so callers don't have to pre-translate.
XTTS_SUPPORTED_LANGS = frozenset({
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar",
    "zh", "zh-cn", "ja", "ko", "hu", "hi", "bn", "ta", "te",
})

# How much to skip at the start of the source audio — YouTube intros,
# opening logos, and auto-generated music beds typically fill this window.
DEFAULT_SKIP_INTRO_SEC = 20.0

# Preferred reference clip length. XTTS handles 3-30s; 8-12s is the sweet
# spot for capturing timbre without overfitting to one phrase.
DEFAULT_REF_DURATION_SEC = 10.0
DEFAULT_MIN_REF_DURATION_SEC = 4.0

# Silence threshold for ffmpeg silencedetect. -30 dB is conservative — softer
# voices fall below -25 dB, and we'd rather include a quiet talker than skip them.
SILENCE_THRESHOLD_DB = -30.0
MIN_SILENCE_GAP_SEC = 0.5


@dataclass(frozen=True)
class CloneDiagnosis:
    """Result of a pre-flight check for voice cloning readiness."""
    ok: bool
    reason: str              # human-readable; OK or the first problem found
    has_gpu: bool
    has_torch: bool
    has_coqui: bool
    language_supported: bool


def check_xtts_prerequisites(target_language: str) -> CloneDiagnosis:
    """Check everything required to run XTTS voice cloning. Returns a
    diagnosis with individual flags so the caller can surface specifics."""
    target_base = (target_language or "").split("-")[0].lower()
    lang_ok = target_base in XTTS_SUPPORTED_LANGS or target_language.lower() in XTTS_SUPPORTED_LANGS

    has_torch = False
    has_gpu = False
    try:
        import torch
        has_torch = True
        try:
            has_gpu = bool(torch.cuda.is_available())
        except Exception:
            has_gpu = False
    except ImportError:
        pass

    has_coqui = False
    try:
        import importlib.util
        has_coqui = importlib.util.find_spec("TTS") is not None
    except Exception:
        has_coqui = False

    if not has_torch:
        return CloneDiagnosis(False, "PyTorch not installed — cannot run XTTS.",
                              has_gpu, has_torch, has_coqui, lang_ok)
    if not has_gpu:
        return CloneDiagnosis(False,
                              "No CUDA GPU detected. XTTS on CPU is ~30-60x realtime and "
                              "not viable for real videos. Use a GPU machine or pick a "
                              "non-cloning engine (Sarvam Bulbul v3 / Google Neural2).",
                              has_gpu, has_torch, has_coqui, lang_ok)
    if not has_coqui:
        return CloneDiagnosis(False,
                              "Coqui TTS package not installed. Run: pip install TTS",
                              has_gpu, has_torch, has_coqui, lang_ok)
    if not lang_ok:
        return CloneDiagnosis(False,
                              f"XTTS v2 doesn't support target language '{target_language}'. "
                              f"Supported: {sorted(XTTS_SUPPORTED_LANGS)}",
                              has_gpu, has_torch, has_coqui, lang_ok)

    return CloneDiagnosis(True, "All XTTS prerequisites satisfied.",
                          has_gpu, has_torch, has_coqui, lang_ok)


def _run_ffmpeg_capture(cmd: List[str], timeout_s: int = 60) -> Tuple[int, str]:
    """Run ffmpeg and return (returncode, combined stderr+stdout)."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            encoding="utf-8", errors="replace",
        )
        return proc.returncode, (proc.stderr or "") + (proc.stdout or "")
    except subprocess.TimeoutExpired:
        return -1, "ffmpeg timed out"
    except Exception as e:
        return -1, f"ffmpeg invocation failed: {e}"


def _probe_duration(ffmpeg: str, audio_path: Path) -> float:
    """Best-effort audio duration in seconds; 0.0 on any error."""
    # Reuse the same binary's ffprobe sibling; fall back to ffmpeg -i parsing.
    rc, out = _run_ffmpeg_capture([ffmpeg, "-i", str(audio_path), "-f", "null", "-"], timeout_s=30)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if m:
        h, mm, s = m.groups()
        return int(h) * 3600 + int(mm) * 60 + float(s)
    return 0.0


def _find_non_silent_window(ffmpeg: str, audio_path: Path,
                             skip_intro: float, min_duration: float,
                             target_duration: float) -> Optional[Tuple[float, float]]:
    """Use ffmpeg silencedetect to find a non-silent window after skip_intro.

    Returns (start_seconds, duration_seconds) or None if nothing suitable.
    """
    total = _probe_duration(ffmpeg, audio_path)
    if total <= 0:
        return None

    cmd = [
        ffmpeg, "-nostats", "-hide_banner",
        "-i", str(audio_path),
        "-af", f"silencedetect=noise={SILENCE_THRESHOLD_DB}dB:d={MIN_SILENCE_GAP_SEC}",
        "-f", "null", "-",
    ]
    rc, out = _run_ffmpeg_capture(cmd, timeout_s=120)
    if rc != 0:
        return None

    # Parse silence_start / silence_end markers in order.
    events: List[Tuple[str, float]] = []
    for m in re.finditer(r"silence_(start|end):\s*(-?\d+(?:\.\d+)?)", out):
        events.append((m.group(1), float(m.group(2))))

    # Build a timeline of non-silent (speech) intervals from the silence markers.
    # Start with a speech interval at 0.0; every silence_start ends it; every
    # silence_end starts a new one.
    speech: List[Tuple[float, float]] = []
    cur_start = 0.0
    last_was_end = False
    for kind, t in events:
        if kind == "start":
            if t > cur_start:
                speech.append((cur_start, t))
            last_was_end = False
        elif kind == "end":
            cur_start = t
            last_was_end = True
    # Tail interval up to the end
    if cur_start < total:
        speech.append((cur_start, total))

    # Pick the first speech region that:
    #   - starts at or after skip_intro (if none fit, relax to any)
    #   - lasts >= min_duration
    # Return min(target_duration, region_length) starting from max(region_start, skip_intro).
    def _pick(regions: List[Tuple[float, float]], earliest: float) -> Optional[Tuple[float, float]]:
        for s, e in regions:
            # Clip to earliest
            start = max(s, earliest)
            length = e - start
            if length >= min_duration:
                return (start, min(length, target_duration))
        return None

    return _pick(speech, skip_intro) or _pick(speech, 0.0)


def _measure_rms(ffmpeg: str, audio_path: Path) -> float:
    """Return mean RMS level in dBFS; ~-60 or lower == effectively silent."""
    rc, out = _run_ffmpeg_capture(
        [ffmpeg, "-nostats", "-hide_banner", "-i", str(audio_path),
         "-af", "volumedetect", "-f", "null", "-"],
        timeout_s=30,
    )
    m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", out)
    return float(m.group(1)) if m else -99.0


def pick_reference_audio(
    ffmpeg: str,
    audio_raw: Path,
    work_dir: Path,
    user_ref_dir: Optional[Path] = None,
    target_sample_rate: int = 22050,
    target_duration: float = DEFAULT_REF_DURATION_SEC,
    skip_intro: float = DEFAULT_SKIP_INTRO_SEC,
    report: Optional[Callable[[str, float, str], None]] = None,
) -> Optional[Path]:
    """Produce a 22050 Hz mono reference WAV suitable for XTTS cloning.

    Priority:
      1. If user_ref_dir has *.wav / *.mp3 files → use the first one (converted).
      2. Otherwise use ffmpeg silencedetect on audio_raw to find a clean speech
         window starting at/after skip_intro seconds. Extract target_duration.
      3. If silence detection yields nothing, fall back to first target_duration
         seconds after skip_intro (or from 0 if the clip is short).

    Returns the produced reference path, or None if audio_raw is missing /
    unusable. Output is always at work_dir / 'voice_ref.wav' so downstream
    engines find it by convention.
    """
    ref_out = work_dir / "voice_ref.wav"

    # Option 1: user-provided reference wins outright.
    if user_ref_dir and user_ref_dir.exists():
        user_refs = sorted(list(user_ref_dir.glob("*.wav")) + list(user_ref_dir.glob("*.mp3")))
        if user_refs:
            src = user_refs[0]
            if report:
                report("clone_ref", 0.1, f"Using user-provided reference: {src.name}")
            rc, _ = _run_ffmpeg_capture([
                ffmpeg, "-y", "-i", str(src),
                "-t", str(target_duration),
                "-ar", str(target_sample_rate), "-ac", "1",
                str(ref_out),
            ], timeout_s=60)
            if rc == 0 and ref_out.exists() and ref_out.stat().st_size > 0:
                return ref_out
            if report:
                report("clone_ref", 0.15,
                       f"User reference {src.name} failed to convert — falling back to source extraction")

    if not audio_raw.exists():
        if report:
            report("clone_ref", 0.0, f"No audio_raw.wav at {audio_raw} — cannot build reference")
        return None

    # Option 2: silence-detected clean speech window
    window = _find_non_silent_window(
        ffmpeg, audio_raw,
        skip_intro=skip_intro,
        min_duration=DEFAULT_MIN_REF_DURATION_SEC,
        target_duration=target_duration,
    )

    if window is not None:
        start, length = window
        if report:
            report("clone_ref", 0.2,
                   f"Voice reference: speech window @ {start:.1f}s for {length:.1f}s")
        rc, _ = _run_ffmpeg_capture([
            ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", str(audio_raw),
            "-t", f"{length:.3f}",
            "-ar", str(target_sample_rate), "-ac", "1",
            str(ref_out),
        ], timeout_s=60)
        if rc == 0 and ref_out.exists() and ref_out.stat().st_size > 0:
            # Validate it's not silent
            rms = _measure_rms(ffmpeg, ref_out)
            if rms > -50.0:
                return ref_out
            if report:
                report("clone_ref", 0.25,
                       f"Chosen window was near-silent ({rms:.1f} dB) — falling back to naive slice")

    # Option 3: naive slice (last resort — matches old behavior but past the intro)
    total = _probe_duration(ffmpeg, audio_raw)
    start = skip_intro if total > (skip_intro + target_duration) else 0.0
    length = min(target_duration, max(total - start, 1.0))
    if report:
        report("clone_ref", 0.3,
               f"Voice reference: fallback slice @ {start:.1f}s for {length:.1f}s")
    rc, _ = _run_ffmpeg_capture([
        ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", str(audio_raw),
        "-t", f"{length:.3f}",
        "-ar", str(target_sample_rate), "-ac", "1",
        str(ref_out),
    ], timeout_s=60)
    if rc == 0 and ref_out.exists() and ref_out.stat().st_size > 0:
        return ref_out
    return None


def validate_reference_audio(ffmpeg: str, ref_path: Path) -> Tuple[bool, str]:
    """Validate a reference clip is actually usable for cloning.

    Returns (ok, reason). Fails on missing file, zero bytes, very short,
    or near-silent audio (mean RMS < -50 dB).
    """
    if not ref_path.exists():
        return False, f"Reference missing: {ref_path}"
    if ref_path.stat().st_size == 0:
        return False, f"Reference file is empty: {ref_path}"

    dur = _probe_duration(ffmpeg, ref_path)
    if dur < DEFAULT_MIN_REF_DURATION_SEC:
        return False, (f"Reference too short ({dur:.1f}s). "
                       f"XTTS needs at least {DEFAULT_MIN_REF_DURATION_SEC}s of speech.")

    rms = _measure_rms(ffmpeg, ref_path)
    if rms < -50.0:
        return False, (f"Reference is near-silent ({rms:.1f} dB mean). "
                       f"XTTS will produce garbage. Drop a better WAV into "
                       f"backend/voices/my_voice_refs/ or use a different source video.")

    return True, f"Reference OK ({dur:.1f}s, {rms:.1f} dB)"
