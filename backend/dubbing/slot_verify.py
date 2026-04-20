"""Slot Verify — diagnostic report for audio/video slot alignment.

Three verify modes:
  "dry_run"      — after TTS, before assembly: report only, nothing touched
  "auto_fix"     — report + pad silence / micro-stretch for warn-level drift
  "post_verify"  — after final video built: ffprobe assembled segments vs expected

The report table shows per-segment: original slot, TTS raw, after speedup,
word count, new slot, drift, and status (ok/warn/fail).
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional

from .contracts import Cue, SLOT_DRIFT_WARN_MS, SLOT_DRIFT_FAIL_MS

log = logging.getLogger(__name__)


# ─── Report generation ──────────────────────────────────────────────────

def generate_report(
    cues: List[Cue],
    drift_warn_ms: float = SLOT_DRIFT_WARN_MS,
    drift_fail_ms: float = SLOT_DRIFT_FAIL_MS,
) -> str:
    """Build a human-readable slot fit report table.

    Returns a formatted string suitable for logging or file output.
    """
    if not cues:
        return "SLOT FIT REPORT: No cues to report.\n"

    lines = []
    sep = "\u2500" * 90
    lines.append("")
    lines.append("\u2550" * 90)
    lines.append("  SLOT FIT REPORT")
    lines.append(sep)
    lines.append(
        f"  {'Seg':>4}  {'Original':>9}  {'TTS Raw':>9}  {'@Speed':>8}  "
        f"{'Words':>5}  {'New Slot':>9}  {'Drift':>8}  {'Status':>6}"
    )
    lines.append(sep)

    total_original = 0.0
    total_new = 0.0
    counts = {"ok": 0, "warn": 0, "fail": 0}

    for cue in cues:
        orig = cue.duration
        tts_raw = cue.tts_duration or orig
        # Back-calculate raw TTS before speedup
        raw_before_speedup = tts_raw * cue.audio_speedup if cue.audio_speedup > 1.0 else tts_raw
        new_dur = cue.new_duration
        drift = cue.slot_drift_ms
        status = cue.slot_status or "ok"
        wc = cue.word_count

        icon = {"ok": "\u2713", "warn": "\u26a0", "fail": "\u2717"}.get(status, "?")
        speed_str = f"{cue.audio_speedup:.2f}x" if cue.audio_speedup > 1.0 else "  1.00x"

        lines.append(
            f"  {cue.id:4d}  {orig:8.2f}s  {raw_before_speedup:8.2f}s  {speed_str}  "
            f"{wc:5d}  {new_dur:8.2f}s  {drift:+7.0f}ms  {icon} {status.upper()}"
        )

        total_original += orig
        total_new += new_dur
        counts[status] = counts.get(status, 0) + 1

    lines.append(sep)

    expansion = ((total_new - total_original) / total_original * 100) if total_original > 0 else 0
    speeds = [c.video_speed for c in cues if c.video_speed != 1.0]
    speed_range = (
        f"{min(speeds):.2f}x \u2013 {max(speeds):.2f}x" if speeds else "1.00x (no change)"
    )

    lines.append(
        f"  Total original: {total_original:.2f}s \u2192 New timeline: {total_new:.2f}s "
        f"({expansion:+.1f}%)"
    )
    lines.append(
        f"  Segments OK: {counts['ok']}/{len(cues)} | "
        f"Warnings: {counts['warn']} | Failures: {counts['fail']}"
    )
    from .contracts import MIN_VIDEO_SPEED
    lines.append(f"  Video speed range: {speed_range} (floor: {MIN_VIDEO_SPEED:.2f}x)")
    lines.append("\u2550" * 90)
    lines.append("")

    return "\n".join(lines)


def save_report(
    cues: List[Cue],
    output_dir: Path,
    drift_warn_ms: float = SLOT_DRIFT_WARN_MS,
    drift_fail_ms: float = SLOT_DRIFT_FAIL_MS,
) -> Path:
    """Write report as both .txt (human) and .json (machine) to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Human-readable
    txt_path = output_dir / "slot_fit_report.txt"
    report_text = generate_report(cues, drift_warn_ms, drift_fail_ms)
    txt_path.write_text(report_text, encoding="utf-8")

    # Machine-readable
    json_path = output_dir / "slot_fit_report.json"
    rows = []
    for cue in cues:
        rows.append({
            "id": cue.id,
            "original_start": cue.start,
            "original_end": cue.end,
            "original_dur": round(cue.duration, 4),
            "tts_duration": cue.tts_duration,
            "audio_speedup": cue.audio_speedup,
            "word_count": cue.word_count,
            "new_start": cue.new_start,
            "new_end": cue.new_end,
            "new_dur": round(cue.new_duration, 4),
            "video_speed": cue.video_speed,
            "drift_ms": cue.slot_drift_ms,
            "status": cue.slot_status or "ok",
        })
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("Slot fit report saved to %s and %s", txt_path, json_path)
    return txt_path


# ─── Verify modes ───────────────────────────────────────────────────────

def dry_run(
    cues: List[Cue],
    output_dir: Path,
    drift_warn_ms: float = SLOT_DRIFT_WARN_MS,
    drift_fail_ms: float = SLOT_DRIFT_FAIL_MS,
) -> str:
    """Generate and save report. No modifications to cues or files."""
    report = generate_report(cues, drift_warn_ms, drift_fail_ms)
    save_report(cues, output_dir, drift_warn_ms, drift_fail_ms)
    log.info("Slot verify dry_run complete")
    return report


def auto_fix(
    cues: List[Cue],
    output_dir: Path,
    drift_warn_ms: float = SLOT_DRIFT_WARN_MS,
    drift_fail_ms: float = SLOT_DRIFT_FAIL_MS,
) -> str:
    """Report + attempt to fix warn/fail segments.

    Fixes applied:
      - warn (drift 50-200ms): pad with silence or micro-stretch audio
      - fail (drift >200ms): flag for re-segmentation (cannot auto-fix safely)
    """
    fixed = 0
    for cue in cues:
        if cue.slot_status == "warn" and cue.slot_drift_ms >= 0:
            # New slot is longer than TTS → will be padded with silence at assembly
            cue.qc_flags.append(f"auto_padded:{cue.slot_drift_ms:.0f}ms")
            cue.slot_drift_ms = 0.0
            cue.slot_status = "ok"
            fixed += 1
        elif cue.slot_status == "warn" and cue.slot_drift_ms < 0:
            # New slot shorter than TTS by 50-200ms → micro-stretch audio
            # Record the needed stretch for assembly to apply
            stretch_factor = cue.new_duration / cue.tts_duration if (cue.tts_duration and cue.tts_duration > 0) else 1.0
            cue.qc_flags.append(f"auto_stretch:{stretch_factor:.4f}")
            cue.slot_drift_ms = 0.0
            cue.slot_status = "ok"
            fixed += 1
        elif cue.slot_status == "fail":
            cue.qc_flags.append("needs_resegmentation")
            log.warning("Cue %d drift %.0fms too large for auto-fix", cue.id, cue.slot_drift_ms)

    report = generate_report(cues, drift_warn_ms, drift_fail_ms)
    save_report(cues, output_dir, drift_warn_ms, drift_fail_ms)
    log.info("Slot verify auto_fix complete: %d segments fixed", fixed)
    return report


def post_verify(
    cues: List[Cue],
    assembled_video: Path,
    output_dir: Path,
    ffprobe: str = "ffprobe",
    drift_warn_ms: float = SLOT_DRIFT_WARN_MS,
    drift_fail_ms: float = SLOT_DRIFT_FAIL_MS,
) -> str:
    """After final assembly, verify the output video duration matches expected.

    This is a whole-file check — we compare total expected vs actual duration.
    Per-segment verification requires chapter markers or keyframe analysis.
    """
    from .slot_recompute import probe_duration

    if not cues:
        return "POST-ASSEMBLY VERIFICATION: No cues to verify.\n"

    actual_total = probe_duration(assembled_video, ffprobe)
    expected_total = cues[-1].new_end if cues[-1].new_end is not None else cues[-1].end

    total_drift_ms = (actual_total - expected_total) * 1000

    lines = []
    lines.append("")
    lines.append("\u2550" * 60)
    lines.append("  POST-ASSEMBLY VERIFICATION")
    lines.append("\u2500" * 60)
    lines.append(f"  Expected duration: {expected_total:.2f}s")
    lines.append(f"  Actual duration:   {actual_total:.2f}s")
    lines.append(f"  Total drift:       {total_drift_ms:+.0f}ms")

    if abs(total_drift_ms) < drift_warn_ms:
        lines.append(f"  Status:            \u2713 OK")
    elif abs(total_drift_ms) < drift_fail_ms:
        lines.append(f"  Status:            \u26a0 WARNING")
    else:
        lines.append(f"  Status:            \u2717 DRIFT TOO LARGE")

    lines.append("\u2550" * 60)
    lines.append("")

    report = "\n".join(lines)

    # Append to existing report file
    txt_path = output_dir / "slot_fit_report.txt"
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(report)

    log.info("Post-verify: expected=%.2fs actual=%.2fs drift=%+.0fms",
             expected_total, actual_total, total_drift_ms)
    return report
