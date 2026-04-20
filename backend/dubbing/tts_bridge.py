"""TTS Bridge — export cues for TTS engine or ElevenLabs manual dub format.

Output format per row:
    speaker, start_time, end_time, transcription, translation
    S1, 12.340, 14.920, He opened the gate., उसने दरवाज़ा खोल दिया।

This maps cleanly to ElevenLabs' manual dubbing row format.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import List

from .contracts import Cue


def export_csv(cues: List[Cue], output_path: Path):
    """Export cues as CSV for ElevenLabs or other TTS tools."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker", "start_time", "end_time", "transcription", "translation"])
        for cue in cues:
            row = cue.to_tts_row()
            writer.writerow([
                row["speaker"],
                row["start_time"],
                row["end_time"],
                row["transcription"],
                row["translation"],
            ])


def export_json(cues: List[Cue], output_path: Path):
    """Export full cue data as JSON for debugging/inspection."""
    data = [cue.to_dict() for cue in cues]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_srt(cues: List[Cue], output_path: Path, text_key: str = "text_hi_display",
               use_revised_timeline: bool = False):
    """Export as SRT subtitle file.

    use_revised_timeline: if True and slot recompute has run, use new_start/new_end.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        srt_idx = 1
        for cue in cues:
            text = getattr(cue, text_key, "") or cue.text_hi_fit or cue.text_hi_raw
            if not text.strip():
                continue
            if use_revised_timeline and cue.new_start is not None and cue.new_end is not None:
                start = cue.new_start
                end = cue.new_end
            else:
                start = cue.start
                end = cue.end
            f.write(f"{srt_idx}\n")
            f.write(f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n")
            f.write(f"{text}\n\n")
            srt_idx += 1


def export_tts_segments(cues: List[Cue]) -> List[dict]:
    """Export as pipeline-compatible TTS segment list (for existing pipeline.py).

    If slot recompute has run (new_start/new_end populated), exports the
    revised timeline so assembly uses the correct timing.
    """
    segments = []
    for cue in cues:
        # Use revised timeline if available, else original
        start = cue.new_start if cue.new_start is not None else cue.start
        end = cue.new_end if cue.new_end is not None else cue.end

        segments.append({
            "start": start,
            "end": end,
            "original_start": cue.start,
            "original_end": cue.end,
            "text": cue.text_clean_en,
            "text_translated": cue.text_hi_fit or cue.text_hi_raw,
            "text_original": cue.text_original,
            "speaker_id": cue.speaker,
            "emotion": cue.emotion,
            "protected_terms": cue.protected_terms,
            "_protected_terms": cue.protected_terms,
            "pronunciation_overrides": cue.pronunciation_overrides,
            "qc_flags": cue.qc_flags,
            # Slot recompute data for assembly
            "audio_speedup": cue.audio_speedup,
            "video_speed": cue.video_speed,
            "slot_status": cue.slot_status,
        })
    return segments


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
