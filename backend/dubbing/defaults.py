"""Production defaults — frozen runtime configuration.

ONE default path. ONE fallback per stage. No option sprawl.
Everything else is experimental/disabled.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProductionDefaults:
    """Locked production configuration. Do not change during runtime."""

    # ── ASR ──
    asr_primary: str = "parakeet"        # Text source: best punctuation/capitalization
    asr_timing: str = "whisperx"         # Timing source: best word-level alignment
    asr_fallback: str = "faster-whisper"  # If both fail

    # ── Translation ──
    translation_primary: str = "nllb_polish"  # IndicTrans2 → LLM → Rules
    translation_fallback: str = "google_polish"  # Google → LLM polish

    # ── TTS ──
    tts_primary: str = "chatterbox"      # Best Hindi quality available
    tts_fallback: str = "edge-tts"       # Always available, no GPU needed

    # ── Audio ──
    audio_speed_min: float = 0.95
    audio_speed_max: float = 1.25
    audio_priority: bool = True          # Audio is king, video adapts

    # ── Cue Building ──
    segmenter: str = "dp"                # "dp" (DP optimal) | "sentence" (sentence-first)
    buffer_pct: float = 0.20             # Hindi expansion buffer for sentence segmenter
    max_sentences_per_cue: int = 2       # max sentences per cue (sentence segmenter)
    cue_word_target: tuple = (8, 14)
    cue_word_hard_max: int = 16
    cue_dur_target: tuple = (1.2, 4.5)
    cue_dur_hard_max: float = 5.5
    cue_wps_hard_max: float = 5.2
    cue_max_cpl: int = 42
    cue_max_lines: int = 2

    # ── Glossary ──
    glossary_auto_extract: bool = True
    glossary_protect_names: bool = True
    glossary_protect_ranks: bool = True

    # ── QC ──
    qc_block_on_fail: bool = True        # pre-TTS QC blocks, not just logs
    qc_max_retries: int = 3

    # ── AV Sync Slot Recompute ──
    av_sync_mode: str = "original"       # "original" | "capped" | "audio_first"
    max_audio_speedup: float = 1.30      # cap for "capped" mode
    min_video_speed: float = 0.70        # below this → flag for re-segment
    slot_verify: str = "off"             # "off" | "dry_run" | "auto_fix" | "post_verify"
    slot_drift_warn_ms: float = 50.0     # warning threshold
    slot_drift_fail_ms: float = 200.0    # failure threshold


# Singleton
PRODUCTION = ProductionDefaults()


# ── Experimental flags (disabled in production) ──
EXPERIMENTAL = {
    "cosyvoice": True,       # Available but not default
    "indic_parler": False,   # Needs HF token
    "elevenlabs": False,     # Paid
    "seamless_m4t": False,   # Not stable enough
    "google_roundtrip": False,  # Damages meaning
}
