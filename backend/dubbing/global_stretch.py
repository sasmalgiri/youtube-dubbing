"""Global Stretch — uniform video slowdown to match total TTS audio length.

No per-segment recompute. No silence gaps. No punctuation splits.
Just: total words → TTS at 1.25x → measure overflow → one video speed.

Usage:
    from backend.dubbing.global_stretch import compute_global_stretch

    result = compute_global_stretch(
        tts_wav_dir=Path("work/tts_wavs"),
        original_video_duration=120.0,
        audio_speedup=1.25,
    )
    # result.video_speed  → e.g. 0.87
    # result.total_audio  → e.g. 137.9s
    # result.overflow      → e.g. 17.9s
"""
from __future__ import annotations
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class GlobalStretchResult:
    """Output of compute_global_stretch."""
    total_words: int                # total word count across all segments
    total_tts_raw: float            # sum of raw TTS durations (seconds)
    audio_speedup: float            # applied speedup (e.g. 1.25)
    total_audio: float              # total TTS after speedup (seconds)
    original_video_duration: float  # original video length (seconds)
    overflow: float                 # total_audio - original_video (seconds)
    video_speed: float              # original_video / total_audio (< 1.0 = slower)

    def report(self) -> str:
        lines = []
        lines.append("")
        lines.append("\u2550" * 60)
        lines.append("  GLOBAL STRETCH REPORT")
        lines.append("\u2500" * 60)
        lines.append(f"  Total words:          {self.total_words}")
        lines.append(f"  TTS raw duration:     {self.total_tts_raw:.2f}s")
        lines.append(f"  Audio speedup:        {self.audio_speedup:.2f}x")
        lines.append(f"  Total audio (final):  {self.total_audio:.2f}s")
        lines.append(f"  Original video:       {self.original_video_duration:.2f}s")
        lines.append(f"  Overflow:             {self.overflow:+.2f}s")
        lines.append(f"  Video speed:          {self.video_speed:.4f}x")
        if self.video_speed < 0.70:
            lines.append(f"  \u26a0 WARNING: video speed below 0.70x — may look unnatural")
        elif self.overflow <= 0:
            lines.append(f"  \u2713 Audio fits — no video slowdown needed")
        else:
            lines.append(f"  \u2713 Slow video to {self.video_speed:.2f}x to match audio")
        lines.append("\u2550" * 60)
        lines.append("")
        return "\n".join(lines)


def _probe_duration(path: Path, ffprobe: str = "ffprobe") -> float:
    """Get duration in seconds via ffprobe."""
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    return float(result.stdout.strip())


def _count_words(text: str) -> int:
    """Count words — strip punctuation, just raw word tokens."""
    import re
    # Remove all punctuation, keep only word characters and spaces
    cleaned = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    return len(cleaned.split())


def compute_global_stretch(
    tts_wav_dir: Path,
    original_video_duration: float,
    audio_speedup: float = 1.25,
    segments: Optional[List[dict]] = None,
    wav_pattern: str = "seg_{id:04d}.wav",
    ffprobe: str = "ffprobe",
) -> GlobalStretchResult:
    """Compute the single uniform video speed needed to match total TTS audio.

    Args:
        tts_wav_dir: directory with rendered TTS wav files
        original_video_duration: length of original video in seconds
        audio_speedup: speedup applied to TTS audio (default 1.25x)
        segments: list of segment dicts (need "text_translated" or "text" for word count)
                  If None, word count comes from scanning wav files only.
        wav_pattern: filename pattern with {id} placeholder (0-indexed)
        ffprobe: path to ffprobe binary

    Returns:
        GlobalStretchResult with all computed values
    """
    if audio_speedup <= 0:
        raise ValueError(f"audio_speedup must be positive, got {audio_speedup}")

    # ── 1. Count total words (no segmentation, no punctuation) ─────────
    total_words = 0
    if segments:
        for seg in segments:
            text = seg.get("text_translated") or seg.get("text") or ""
            total_words += _count_words(text)

    # ── 2. Measure total TTS raw duration ──────────────────────────────
    total_tts_raw = 0.0
    wav_files = sorted(tts_wav_dir.glob("*.wav"))

    if not wav_files:
        # Try pattern-based discovery
        if segments:
            for i in range(len(segments)):
                wav_name = wav_pattern.format(id=i)
                wav_path = tts_wav_dir / wav_name
                if wav_path.exists():
                    wav_files.append(wav_path)

    for wav_path in wav_files:
        try:
            dur = _probe_duration(wav_path, ffprobe)
            total_tts_raw += dur
        except Exception as e:
            log.warning("Failed to probe %s: %s", wav_path, e)

    if total_tts_raw == 0:
        raise RuntimeError(f"No TTS wavs found or all probes failed in {tts_wav_dir}")

    # ── 3. Apply speedup ───────────────────────────────────────────────
    total_audio = total_tts_raw / audio_speedup

    # ── 4. Measure overflow ────────────────────────────────────────────
    overflow = total_audio - original_video_duration

    # ── 5. Compute uniform video speed ─────────────────────────────────
    if overflow > 0:
        # Audio is longer → slow video down
        video_speed = original_video_duration / total_audio
    else:
        # Audio fits or is shorter → no slowdown needed
        video_speed = 1.0

    result = GlobalStretchResult(
        total_words=total_words,
        total_tts_raw=round(total_tts_raw, 4),
        audio_speedup=audio_speedup,
        total_audio=round(total_audio, 4),
        original_video_duration=round(original_video_duration, 4),
        overflow=round(overflow, 4),
        video_speed=round(video_speed, 4),
    )

    log.info("Global stretch: raw=%.1fs @%.2fx=%.1fs video=%.1fs → speed=%.4f",
             total_tts_raw, audio_speedup, total_audio,
             original_video_duration, video_speed)

    return result


def save_report(result: GlobalStretchResult, output_dir: Path) -> Path:
    """Save the report as txt + json."""
    import json
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / "global_stretch_report.txt"
    txt_path.write_text(result.report(), encoding="utf-8")

    json_path = output_dir / "global_stretch_report.json"
    json_path.write_text(json.dumps(result.__dict__, indent=2), encoding="utf-8")

    return txt_path
