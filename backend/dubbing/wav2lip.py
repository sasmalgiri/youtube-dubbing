"""Wav2Lip lip-sync post-processing wrapper.

Wav2Lip is NOT shipped with this repo (the model + code is ~500MB). User must:
  1. Clone https://github.com/Rudrabha/Wav2Lip into backend/wav2lip/
  2. Download wav2lip_gan.pth into backend/wav2lip/checkpoints/
  3. pip install -r backend/wav2lip/requirements.txt
  4. Toggle use_wav2lip=True in DubbingSettings

Call apply_lipsync() after the dubbed MP4 is assembled. On any failure
(missing repo, missing checkpoint, no face detected, subprocess error)
returns False — the caller keeps the original assembled video.

Wav2Lip performs best on:
  - Single front-facing speaker in a close-up shot
  - Clean audio with clear speech (our dubbed Hindi qualifies)
  - GPU runtime (CPU is ~10-20× realtime, impractical for real videos)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

_WAV2LIP_REPO_DIR = Path(__file__).resolve().parent.parent / "wav2lip"
_CHECKPOINT_DIR = _WAV2LIP_REPO_DIR / "checkpoints"


def _pick_checkpoint() -> Optional[Path]:
    """Prefer wav2lip_gan.pth (better quality); fall back to wav2lip.pth."""
    for name in ("wav2lip_gan.pth", "wav2lip.pth"):
        p = _CHECKPOINT_DIR / name
        if p.exists():
            return p
    return None


def is_available() -> bool:
    """True when the Wav2Lip repo and at least one checkpoint are present."""
    return (
        _WAV2LIP_REPO_DIR.exists()
        and (_WAV2LIP_REPO_DIR / "inference.py").exists()
        and _pick_checkpoint() is not None
    )


def availability_reason() -> str:
    """Human-readable explanation of why is_available() is False."""
    if not _WAV2LIP_REPO_DIR.exists():
        return (
            f"Wav2Lip repo not found at {_WAV2LIP_REPO_DIR}. "
            f"Clone https://github.com/Rudrabha/Wav2Lip into that path."
        )
    if not (_WAV2LIP_REPO_DIR / "inference.py").exists():
        return f"Wav2Lip repo at {_WAV2LIP_REPO_DIR} is missing inference.py."
    if _pick_checkpoint() is None:
        return (
            f"No Wav2Lip checkpoint in {_CHECKPOINT_DIR}. "
            f"Download wav2lip_gan.pth (recommended) or wav2lip.pth."
        )
    return "Wav2Lip is available."


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def apply_lipsync(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    work_dir: Path,
    checkpoint: Optional[Path] = None,
    report: Optional[Callable[[str, float, str], None]] = None,
    timeout_seconds: int = 3600,
    nosmooth: bool = True,
) -> bool:
    """Run Wav2Lip inference to re-sync the video's lips to the given audio.

    Returns True iff a valid output file was produced at output_path.
    On any failure, returns False and the caller keeps the original video.

    Args:
        video_path: input MP4 (dubbed, pre-lipsync)
        audio_path: dubbed audio (typically the same track already muxed in)
        output_path: where the lip-synced MP4 will be written
        work_dir: scratch directory (unused today, reserved)
        checkpoint: optional override; defaults to wav2lip_gan.pth if present
        report: optional callback(stage, progress, message)
        timeout_seconds: subprocess hard cap (default 1 hour)
        nosmooth: pass --nosmooth to inference.py (recommended for talking heads)
    """
    if not is_available():
        if report:
            report("lipsync", 0.0, availability_reason())
        return False

    if not video_path.exists():
        if report:
            report("lipsync", 0.0, f"Wav2Lip: input video not found at {video_path}")
        return False
    if not audio_path.exists():
        if report:
            report("lipsync", 0.0, f"Wav2Lip: audio not found at {audio_path}")
        return False

    ckpt = checkpoint or _pick_checkpoint()
    if ckpt is None:
        if report:
            report("lipsync", 0.0, availability_reason())
        return False

    gpu = _has_gpu()
    if not gpu and report:
        report("lipsync", 0.05,
               "Wav2Lip: no GPU detected — CPU inference is ~10-20× realtime. "
               "Consider disabling use_wav2lip for long videos.")

    if report:
        report("lipsync", 0.10,
               f"Wav2Lip: applying lip sync with {ckpt.name} "
               f"({'GPU' if gpu else 'CPU'})...")

    cmd = [
        sys.executable,
        str(_WAV2LIP_REPO_DIR / "inference.py"),
        "--checkpoint_path", str(ckpt),
        "--face", str(video_path),
        "--audio", str(audio_path),
        "--outfile", str(output_path),
    ]
    if nosmooth:
        cmd.append("--nosmooth")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_WAV2LIP_REPO_DIR),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        if report:
            report("lipsync", 1.0,
                   f"Wav2Lip timed out after {timeout_seconds // 60} min — keeping original video")
        return False
    except FileNotFoundError as e:
        if report:
            report("lipsync", 1.0, f"Wav2Lip subprocess not launchable: {e}")
        return False
    except Exception as e:
        if report:
            report("lipsync", 1.0, f"Wav2Lip subprocess error: {e}")
        return False

    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout or "")[-500:]
        if report:
            report("lipsync", 1.0,
                   f"Wav2Lip exited with code {proc.returncode}. Tail: {err_tail.strip()}")
        return False

    if not output_path.exists() or output_path.stat().st_size == 0:
        if report:
            report("lipsync", 1.0, "Wav2Lip produced no output file — keeping original")
        return False

    if report:
        report("lipsync", 1.0, "Wav2Lip: lip sync applied successfully")
    return True
