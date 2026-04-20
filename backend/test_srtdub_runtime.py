"""Runtime end-to-end verification of srtdub.

Generates synthetic videos via ffmpeg and runs each pipeline stage for real:
  - TTS (real Edge-TTS calls)
  - PCM concat (real WAV writes)
  - Stretch (real ffmpeg setpts)
  - Trim (real ffmpeg -t)
  - Freeze-pad (real ffmpeg tpad)
  - Mux (real audio+video mux)
  - Final mp4 validation (probe, plays, has audio + video streams)

No network video download — uses ffmpeg-generated test patterns.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dubbing.srtdub import (
    _resolve_executable,
    _parse_srt,
    _tts_all_cues,
    _concat_wavs_zero_gap,
    _stretch_and_extend,
    _get_duration,
)

ffmpeg = _resolve_executable("ffmpeg")
ffprobe = ffmpeg.replace("ffmpeg", "ffprobe") if "ffmpeg" in ffmpeg else "ffprobe"
print(f"[setup] ffmpeg = {ffmpeg}")
print(f"[setup] ffprobe = {ffprobe}")

WORK = Path(tempfile.mkdtemp(prefix="srtdub_test_"))
print(f"[setup] work dir = {WORK}")

PASS, FAIL = [], []


def chk(name: str, cond: bool, detail: str = ""):
    if cond:
        print(f"  [PASS] {name}  {detail}")
        PASS.append(name)
    else:
        print(f"  [FAIL] {name}  {detail}")
        FAIL.append(name)


def make_test_video(out_path: Path, duration: float, fps: int = 24) -> None:
    """ffmpeg-generated colour test pattern."""
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi",
         "-i", f"testsrc=size=320x240:rate={fps}:duration={duration}",
         "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p",
         str(out_path)],
        check=True, capture_output=True)


def probe(path: Path) -> dict:
    """ffprobe → JSON dict of streams + format."""
    r = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, timeout=15)
    return json.loads(r.stdout) if r.stdout else {}


# ─────────────────────────────────────────────────────────────────────────
# Test 1: SRT parser already passes 13/13 in static tests — skip
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# Test 2: Real Edge-TTS on short Hindi cues
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Test 2: Edge-TTS on real Hindi cues ===")
cues = [
    {"idx": 1, "start": 0.0, "end": 3.0, "text": "नमस्ते दुनिया।"},
    {"idx": 2, "start": 3.0, "end": 6.0, "text": "यह एक परीक्षण है।"},
    {"idx": 3, "start": 6.0, "end": 9.0, "text": "तीसरी पंक्ति।"},
]
cues = _tts_all_cues(cues, WORK, "hi-IN-SwaraNeural", "+0%",
                      ffmpeg, lambda *a: None, lambda: False)
chk("all 3 cues produced WAVs",
    all(c.get("wav") and Path(c["wav"]).exists() for c in cues),
    f"  → {[Path(c['wav']).name if c.get('wav') else 'None' for c in cues]}")
for c in cues:
    if c.get("wav"):
        info = probe(Path(c["wav"]))
        sr = next((s.get("sample_rate") for s in info.get("streams", [])
                   if s.get("codec_type") == "audio"), None)
        ch = next((s.get("channels") for s in info.get("streams", [])
                   if s.get("codec_type") == "audio"), None)
        codec = next((s.get("codec_name") for s in info.get("streams", [])
                      if s.get("codec_type") == "audio"), None)
        chk(f"WAV cue {c['idx']} format", sr == "48000" and ch == 2 and codec == "pcm_s16le",
            f"  sr={sr} ch={ch} codec={codec}")

# ─────────────────────────────────────────────────────────────────────────
# Test 3: Zero-gap concat
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Test 3: Zero-gap audio concat ===")
timeline = WORK / "timeline.wav"
audio_dur = _concat_wavs_zero_gap(cues, timeline)
chk("timeline file written", timeline.exists())
chk("timeline has positive duration", audio_dur > 0, f"  audio_dur = {audio_dur:.2f}s")
probed_dur = _get_duration(timeline, ffmpeg)
chk("probed duration matches computed",
    abs(probed_dur - audio_dur) < 0.05,
    f"  probed={probed_dur:.2f}s computed={audio_dur:.2f}s")
# Verify: concat duration should be SUM of individual cue durations (zero gap)
sum_dur = sum(_get_duration(Path(c["wav"]), ffmpeg) for c in cues if c.get("wav"))
chk("concat = sum of parts (zero gap proven)",
    abs(audio_dur - sum_dur) < 0.05,
    f"  parts_sum={sum_dur:.2f}s timeline={audio_dur:.2f}s")

# ─────────────────────────────────────────────────────────────────────────
# Test 4: Stretch regime — audio LONGER than video
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Test 4: Stretch regime (audio > video) ===")
short_video = WORK / "short.mp4"
make_test_video(short_video, duration=2.0)
target = audio_dur  # ~6s probably, way longer than 2s
out = WORK / "stretched.mp4"
_stretch_and_extend(short_video, target, max_stretch=10.0, out_path=out, ffmpeg=ffmpeg)
chk("stretched file exists", out.exists())
chk("source video.mp4 NOT touched (still 2s)",
    abs(_get_duration(short_video, ffmpeg) - 2.0) < 0.1,
    f"  source still {_get_duration(short_video, ffmpeg):.2f}s")
out_dur = _get_duration(out, ffmpeg)
chk("output duration matches target audio",
    abs(out_dur - target) < 0.2,
    f"  out={out_dur:.2f}s target={target:.2f}s")

# ─────────────────────────────────────────────────────────────────────────
# Test 5: Stretch + freeze-pad regime — audio FAR longer than even max stretch
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Test 5: Stretch + freeze-pad (audio > video × max_stretch) ===")
tiny_video = WORK / "tiny.mp4"
make_test_video(tiny_video, duration=1.0)
out2 = WORK / "stretched_padded.mp4"
# 1s video, 6s audio, max 2x stretch → 2s stretched + 4s freeze-pad = 6s
_stretch_and_extend(tiny_video, audio_dur, max_stretch=2.0, out_path=out2, ffmpeg=ffmpeg)
chk("stretch+pad file exists", out2.exists())
out2_dur = _get_duration(out2, ffmpeg)
chk("output duration matches target (after pad)",
    abs(out2_dur - audio_dur) < 0.3,
    f"  out={out2_dur:.2f}s target={audio_dur:.2f}s")

# ─────────────────────────────────────────────────────────────────────────
# Test 6: Trim regime — audio SHORTER than video
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Test 6: Trim regime (audio < video) ===")
long_video = WORK / "long.mp4"
make_test_video(long_video, duration=20.0)
out3 = WORK / "trimmed.mp4"
_stretch_and_extend(long_video, audio_dur, max_stretch=10.0, out_path=out3, ffmpeg=ffmpeg)
chk("trimmed file exists", out3.exists())
chk("source video.mp4 NOT touched (still 20s)",
    abs(_get_duration(long_video, ffmpeg) - 20.0) < 0.1)
out3_dur = _get_duration(out3, ffmpeg)
chk("output trimmed to audio duration",
    abs(out3_dur - audio_dur) < 0.2,
    f"  out={out3_dur:.2f}s target={audio_dur:.2f}s")

# ─────────────────────────────────────────────────────────────────────────
# Test 7: Near-equal regime (stream-copy fast path)
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Test 7: Near-equal regime (stream-copy) ===")
exact_video = WORK / "exact.mp4"
make_test_video(exact_video, duration=audio_dur)
out4 = WORK / "exact_out.mp4"
_stretch_and_extend(exact_video, audio_dur, max_stretch=10.0, out_path=out4, ffmpeg=ffmpeg)
chk("near-equal file exists", out4.exists())
chk("source video.mp4 NOT touched",
    abs(_get_duration(exact_video, ffmpeg) - audio_dur) < 0.2)

# ─────────────────────────────────────────────────────────────────────────
# Test 8: Final mux — audio + adapted video
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Test 8: Mux final mp4 ===")
final = WORK / "final.mp4"
final.unlink(missing_ok=True)
mux = subprocess.run(
    [ffmpeg, "-y", "-i", str(out), "-i", str(timeline),
     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
     "-map", "0:v:0", "-map", "1:a:0",
     "-movflags", "+faststart",
     str(final)],
    capture_output=True, text=True, timeout=120)
chk("mux returned 0", mux.returncode == 0,
    f"  stderr tail: {(mux.stderr or '')[-200:].strip()}" if mux.returncode != 0 else "")
chk("final mp4 exists and >1KB",
    final.exists() and final.stat().st_size > 1024,
    f"  size = {final.stat().st_size if final.exists() else 0} bytes")
info = probe(final)
streams = info.get("streams", [])
v_streams = [s for s in streams if s.get("codec_type") == "video"]
a_streams = [s for s in streams if s.get("codec_type") == "audio"]
chk("final mp4 has video stream", len(v_streams) >= 1,
    f"  codec={v_streams[0].get('codec_name') if v_streams else None}")
chk("final mp4 has audio stream", len(a_streams) >= 1,
    f"  codec={a_streams[0].get('codec_name') if a_streams else None}")
final_dur = _get_duration(final, ffmpeg)
chk("final duration ≈ audio duration",
    abs(final_dur - audio_dur) < 0.5,
    f"  final={final_dur:.2f}s audio={audio_dur:.2f}s")

# ─────────────────────────────────────────────────────────────────────────
# Test 9: SRT → cues round-trip
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Test 9: SRT → cues round-trip ===")
srt = """1
00:00:01,000 --> 00:00:03,000
नमस्ते

2
00:00:03,500 --> 00:00:06,000
दुनिया
"""
parsed = _parse_srt(srt)
chk("parsed 2 cues", len(parsed) == 2, f"  got {len(parsed)}")
chk("cue 1 text correct", parsed[0]["text"] == "नमस्ते")
chk("cue 1 timestamps correct",
    parsed[0]["start"] == 1.0 and parsed[0]["end"] == 3.0,
    f"  start={parsed[0]['start']} end={parsed[0]['end']}")

# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"PASSED: {len(PASS)}")
print(f"FAILED: {len(FAIL)}")
if FAIL:
    print("\nFailed tests:")
    for f in FAIL:
        print(f"  - {f}")
print()
print(f"Work dir (intermediate files): {WORK}")
print()
sys.exit(0 if not FAIL else 1)
