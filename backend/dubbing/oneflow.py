"""OneFlow Pipeline — fastest possible dubbing path.

Groq Whisper → Google Translate (parallel) → quality check → re-translate bad ones
→ Edge-TTS (parallel) → correction check → re-TTS bad ones
→ fixed 1.15x speed → video adapts

NO LLM polish. NO cue rebuilding. NO duration matching.
Just: transcribe → translate → speak → assemble.

Speed targets:
- Google Translate: 100 parallel workers (adaptive)
- Edge-TTS: 150 parallel workers (adaptive)
- One quality check + one retry per stage (not more)
"""
from __future__ import annotations
import asyncio
import os
import re
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Callable, Optional
from concurrent.futures import ThreadPoolExecutor

from .worker_manager import WorkerManager

# Fixed audio speed for entire file
ONEFLOW_SPEED = 1.15


def run_oneflow(
    source_url: str,
    work_dir: Path,
    output_path: Path,
    target_language: str = "hi",
    tts_voice: str = "hi-IN-SwaraNeural",
    tts_rate: str = "+0%",
    audio_bitrate: str = "320k",
    on_progress: Callable = None,
    cancel_check: Callable = None,
    ffmpeg: str = "ffmpeg",
    ytdlp: str = "yt-dlp",
) -> Path:
    """Run the OneFlow pipeline end-to-end.

    Returns path to output MP4.
    """
    progress = on_progress or (lambda *_: None)
    is_cancelled = cancel_check or (lambda: False)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mgr = WorkerManager(max_edge=150, max_google=100)

    # ══════════════════════════════════════════════════
    # STEP 1: Download
    # ══════════════════════════════════════════════════
    progress("download", 0.0, "Downloading video...")
    video_path = work_dir / "video.mp4"
    audio_raw = work_dir / "audio_raw.wav"

    if not video_path.exists():
        cmd = [ytdlp, "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
               "--merge-output-format", "mp4", "-o", str(video_path), source_url]
        subprocess.run(cmd, capture_output=True, timeout=300)

    if not video_path.exists():
        cmd = [ytdlp, "-f", "best", "-o", str(video_path), source_url]
        subprocess.run(cmd, capture_output=True, timeout=300)

    if not video_path.exists():
        raise RuntimeError(f"Failed to download video from {source_url}")

    progress("download", 1.0, "Downloaded")

    # ══════════════════════════════════════════════════
    # STEP 2: Extract audio
    # ══════════════════════════════════════════════════
    progress("extract", 0.0, "Extracting audio...")
    if not audio_raw.exists():
        subprocess.run(
            [ffmpeg, "-y", "-i", str(video_path), "-vn",
             "-ar", "48000", "-ac", "2", "-acodec", "pcm_s16le", str(audio_raw)],
            check=True, capture_output=True)
    if not audio_raw.exists():
        raise RuntimeError("Failed to extract audio from video")
    progress("extract", 1.0, "Audio extracted")

    if is_cancelled():
        return output_path

    # ══════════════════════════════════════════════════
    # STEP 3: Transcribe with Groq Whisper (fast cloud)
    # ══════════════════════════════════════════════════
    progress("transcribe", 0.0, "Groq Whisper: transcribing...")
    segments = _transcribe_groq(audio_raw, work_dir, progress, ffmpeg)
    if not segments:
        raise RuntimeError("Groq transcription returned no segments — check audio quality")
    progress("transcribe", 1.0, f"Transcribed: {len(segments)} segments")

    if is_cancelled():
        return output_path

    # ══════════════════════════════════════════════════
    # STEP 4: Google Translate (parallel, adaptive workers)
    # ══════════════════════════════════════════════════
    progress("translate", 0.0, f"Google Translate: {len(segments)} segments ({mgr.get_google_workers()} workers)...")
    segments = _translate_google_parallel(segments, target_language, mgr, progress)

    # Quality check: flag bad translations
    bad_segs = []
    for seg in segments:
        hi = seg.get("text_translated", "")
        en = seg.get("text", "")
        if not hi or hi == en or (len(hi) < len(en) * 0.3 and len(en) > 10):
            bad_segs.append(seg)

    # One re-translate for bad ones (no more retries after this)
    if bad_segs:
        progress("translate", 0.85, f"Re-translating {len(bad_segs)} bad segments...")
        _translate_google_parallel(bad_segs, target_language, mgr, progress, is_retry=True)

    # Validate: at least some translations exist
    translated_count = sum(1 for s in segments if s.get("text_translated", "") != s.get("text", ""))
    if translated_count < len(segments) * 0.3:
        raise RuntimeError(f"Translation failed: only {translated_count}/{len(segments)} segments translated")

    progress("translate", 1.0, f"Translated: {translated_count}/{len(segments)} segments")

    if is_cancelled():
        return output_path

    # ══════════════════════════════════════════════════
    # STEP 5: Edge-TTS (parallel, adaptive workers)
    # ══════════════════════════════════════════════════
    progress("synthesize", 0.0, f"Edge-TTS: {len(segments)} segments ({mgr.get_edge_workers()} workers)...")
    tts_data = _tts_edge_parallel(segments, work_dir, tts_voice, tts_rate, mgr, progress, ffmpeg)

    if not tts_data:
        raise RuntimeError("All TTS synthesis failed — no audio generated")

    # Correction check: flag bad TTS (missing/tiny WAV)
    bad_tts_indices = []
    for i, tts in enumerate(tts_data):
        if not tts["wav"].exists() or tts["wav"].stat().st_size < 1000 or tts["duration"] < 0.2:
            bad_tts_indices.append(i)

    # One re-TTS for bad ones (no more retries after this)
    if bad_tts_indices:
        progress("synthesize", 0.9, f"Re-synthesizing {len(bad_tts_indices)} bad segments...")
        for idx in bad_tts_indices:
            if idx >= len(segments):
                continue
            text = segments[idx].get("text_translated", "")
            if not text:
                continue
            wav = work_dir / f"tts_retry_{idx:04d}.wav"
            try:
                _edge_tts_single(text, wav, tts_voice, tts_rate, ffmpeg)
                if wav.exists() and wav.stat().st_size > 1000:
                    tts_data[idx]["wav"] = wav
                    tts_data[idx]["duration"] = _get_duration(wav, ffmpeg)
            except Exception:
                pass

    progress("synthesize", 1.0, f"TTS done: {len(tts_data)} segments")

    if is_cancelled():
        return output_path

    # ══════════════════════════════════════════════════
    # STEP 6: Fixed 1.15x speed on ALL segments
    # ══════════════════════════════════════════════════
    progress("assemble", 0.0, f"Applying fixed {ONEFLOW_SPEED}x speed to all audio...")

    def _speed_one(args):
        i, tts = args
        if not tts["wav"].exists():
            return
        sped = work_dir / f"tts_sped_{i:04d}.wav"
        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", str(tts["wav"]),
                 "-filter:a", f"atempo={ONEFLOW_SPEED}",
                 "-ar", "48000", "-ac", "2", str(sped)],
                check=True, capture_output=True)
            tts["wav"] = sped
            tts["duration"] = tts["duration"] / ONEFLOW_SPEED
        except Exception:
            pass  # Keep original speed if atempo fails

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(_speed_one, enumerate(tts_data)))

    progress("assemble", 0.15, "Speed applied")

    # ══════════════════════════════════════════════════
    # STEP 7: Build audio timeline + assemble video
    # ══════════════════════════════════════════════════
    progress("assemble", 0.2, "Building audio timeline...")
    video_duration = _get_duration(video_path, ffmpeg)
    if video_duration <= 0:
        raise RuntimeError("Cannot determine video duration")

    timeline_wav = work_dir / "oneflow_timeline.wav"
    timeline_wav.parent.mkdir(parents=True, exist_ok=True)
    _build_timeline_inmemory(tts_data, video_duration, timeline_wav)

    progress("assemble", 0.5, "Timeline built")

    # Clean previous output
    if output_path.exists():
        output_path.unlink()

    progress("assemble", 0.6, "Muxing video + audio...")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(video_path), "-i", str(timeline_wav),
         "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
         "-shortest", str(output_path)],
        check=True, capture_output=True)

    if not output_path.exists():
        raise RuntimeError("Final muxing failed — output file not created")

    progress("assemble", 0.9, "Cleaning up...")

    # Cleanup temp files
    for pattern in ["tts_sped_*.wav", "tts_retry_*.wav", "tts_of_*.wav", "tts_of_*.mp3"]:
        for f in work_dir.glob(pattern):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

    progress("assemble", 1.0, "Done!")
    return output_path


# ── Helper functions ──────────────────────────────────────────────────


def _transcribe_groq(audio_raw: Path, work_dir: Path,
                     progress: Callable, ffmpeg: str) -> List[Dict]:
    """Transcribe with Groq Whisper API (fastest cloud ASR)."""
    # Load all Groq keys for rotation
    groq_keys = [os.environ.get("GROQ_API_KEY", "").strip()]
    for i in range(2, 20):
        k = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
        if k:
            groq_keys.append(k)
    groq_keys = [k for k in groq_keys if k]
    if not groq_keys:
        raise RuntimeError("No GROQ_API_KEY set — required for OneFlow")
    _groq_idx = [0]
    def _next_groq_key():
        key = groq_keys[_groq_idx[0] % len(groq_keys)]
        _groq_idx[0] += 1
        return key
    groq_key = _next_groq_key()

    import requests

    # Groq needs <25MB file, so compress to smaller format
    audio_small = work_dir / "audio_groq.m4a"
    if not audio_small.exists():
        subprocess.run(
            [ffmpeg, "-y", "-i", str(audio_raw),
             "-ar", "16000", "-ac", "1", "-b:a", "32k", str(audio_small)],
            check=True, capture_output=True)

    # Check file size — Groq limit is 25MB
    file_size = audio_small.stat().st_size
    if file_size > 25 * 1024 * 1024:
        # Split into chunks
        return _transcribe_groq_chunked(audio_small, work_dir, groq_key, progress, ffmpeg)

    progress("transcribe", 0.3, "Sending to Groq Whisper...")
    with open(audio_small, "rb") as f:
        resp = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {groq_key}"},
            files={"file": (audio_small.name, f, "audio/m4a")},
            data={
                "model": "whisper-large-v3",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
                "language": "en",
            },
            timeout=300,
        )
    resp.raise_for_status()
    data = resp.json()

    segments = []
    for seg in data.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            segments.append({
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": text,
            })

    return segments


def _transcribe_groq_chunked(audio_path: Path, work_dir: Path,
                              groq_key: str, progress: Callable,
                              ffmpeg: str) -> List[Dict]:
    """Split audio into 10-min chunks and transcribe each with Groq."""
    import requests

    chunk_dir = work_dir / "groq_chunks"
    chunk_dir.mkdir(exist_ok=True)

    # Get duration
    dur = _get_duration(audio_path, ffmpeg)
    chunk_duration = 600  # 10 minutes per chunk
    chunks = []

    for start in range(0, int(dur), chunk_duration):
        chunk_path = chunk_dir / f"chunk_{start:06d}.m4a"
        if not chunk_path.exists():
            subprocess.run(
                [ffmpeg, "-y", "-i", str(audio_path),
                 "-ss", str(start), "-t", str(chunk_duration),
                 "-ar", "16000", "-ac", "1", "-b:a", "32k", str(chunk_path)],
                check=True, capture_output=True)
        chunks.append((start, chunk_path))

    all_segments = []
    for i, (offset, chunk_path) in enumerate(chunks):
        progress("transcribe", 0.1 + 0.8 * (i / len(chunks)),
                 f"Groq Whisper: chunk {i + 1}/{len(chunks)}...")

        with open(chunk_path, "rb") as f:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {groq_key}"},
                files={"file": (chunk_path.name, f, "audio/m4a")},
                data={
                    "model": "whisper-large-v3",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                    "language": "en",
                },
                timeout=300,
            )
        resp.raise_for_status()
        data = resp.json()

        for seg in data.get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                all_segments.append({
                    "start": float(seg.get("start", 0)) + offset,
                    "end": float(seg.get("end", 0)) + offset,
                    "text": text,
                })

    return all_segments


def _translate_google_parallel(segments: List[Dict], target_lang: str,
                                mgr: WorkerManager, progress: Callable,
                                is_retry: bool = False) -> List[Dict]:
    """Translate all segments with Google Translate using adaptive parallel workers."""
    from deep_translator import GoogleTranslator

    total = len(segments)
    done = [0]
    lock = __import__("threading").Lock()

    def _is_garbage(text):
        if not text:
            return True
        low = text.lower()
        return ("error 500" in low or "server error" in low
                or "that's an error" in low or "<html" in low
                or "<!doctype" in low)

    def _translate_one(seg):
        text = seg.get("text", "").strip()
        if not text:
            return
        for _attempt in range(3):
            try:
                translator = GoogleTranslator(source='auto', target=target_lang)
                result = translator.translate(text)
                if result and not _is_garbage(result):
                    seg["text_translated"] = result
                    mgr.report_success("google")
                    return
                # Garbage response — retry
                mgr.report_failure("google")
                import time; time.sleep(1.5 * (_attempt + 1))
            except Exception:
                mgr.report_failure("google")
                import time; time.sleep(1.5 * (_attempt + 1))
        # All retries failed — keep original
        seg.setdefault("text_translated", text)

        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                progress("translate", 0.1 + 0.7 * (done[0] / total),
                         f"Google Translate: {done[0]}/{total} ({mgr.get_google_workers()} workers)...")

    workers = mgr.get_google_workers()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_translate_one, segments))

    return segments


def _tts_edge_parallel(segments: List[Dict], work_dir: Path,
                       voice: str, rate: str, mgr: WorkerManager,
                       progress: Callable, ffmpeg: str) -> List[Dict]:
    """Generate TTS for all segments with Edge-TTS using adaptive parallel workers."""
    import edge_tts

    total = len(segments)
    tts_data = [None] * total
    done = [0]
    lock = __import__("threading").Lock()

    async def _generate_one(i, seg, semaphore):
        text = seg.get("text_translated", seg.get("text", "")).strip()
        if not text:
            return

        async with semaphore:
            mp3 = work_dir / f"tts_of_{i:04d}.mp3"
            wav = work_dir / f"tts_of_{i:04d}.wav"
            try:
                comm = edge_tts.Communicate(text, voice, rate=rate)
                await comm.save(str(mp3))

                if mp3.exists() and mp3.stat().st_size > 0:
                    subprocess.run(
                        [ffmpeg, "-y", "-i", str(mp3),
                         "-ar", "48000", "-ac", "2", str(wav)],
                        check=True, capture_output=True)
                    mp3.unlink(missing_ok=True)

                    dur = _get_duration(wav, ffmpeg)
                    tts_data[i] = {
                        "start": seg["start"],
                        "end": seg["end"],
                        "wav": wav,
                        "duration": dur,
                    }
                    mgr.report_success("edge")
                else:
                    mgr.report_failure("edge")
            except Exception:
                mgr.report_failure("edge")

            with lock:
                done[0] += 1
                if done[0] % 30 == 0:
                    progress("synthesize", 0.1 + 0.8 * (done[0] / total),
                             f"Edge-TTS: {done[0]}/{total} ({mgr.get_edge_workers()} workers)...")

    async def _run_all():
        workers = mgr.get_edge_workers()
        semaphore = asyncio.Semaphore(workers)
        tasks = [_generate_one(i, seg, semaphore) for i, seg in enumerate(segments)]
        await asyncio.gather(*tasks)

    asyncio.run(_run_all())

    # Filter out None entries (failed segments)
    return [t for t in tts_data if t is not None]


def _edge_tts_single(text: str, wav_path: Path, voice: str, rate: str, ffmpeg: str):
    """Generate single TTS segment with Edge-TTS."""
    import edge_tts

    mp3 = wav_path.with_suffix(".mp3")

    async def _gen():
        comm = edge_tts.Communicate(text, voice, rate=rate)
        await comm.save(str(mp3))

    asyncio.run(_gen())

    if mp3.exists() and mp3.stat().st_size > 0:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(mp3),
             "-ar", "48000", "-ac", "2", str(wav_path)],
            check=True, capture_output=True)
        mp3.unlink(missing_ok=True)


def _build_timeline_inmemory(tts_data: List[Dict], total_duration: float,
                              output_path: Path, sample_rate: int = 48000):
    """Build audio timeline using in-memory bytearray (fastest method)."""
    import struct

    n_channels = 2
    total_samples = int((total_duration + 1.0) * sample_rate)
    # Allocate silence buffer
    timeline = bytearray(total_samples * n_channels * 2)  # 16-bit samples

    for tts in tts_data:
        wav_path = tts["wav"]
        if not wav_path.exists():
            continue

        start_sample = int(tts["start"] * sample_rate)
        if start_sample < 0 or start_sample >= total_samples:
            continue

        # Read WAV data — properly parse header to find data chunk
        try:
            with open(wav_path, "rb") as f:
                riff = f.read(12)
                if riff[:4] != b'RIFF' or riff[8:12] != b'WAVE':
                    continue
                # Find 'data' chunk (skip fmt, LIST, etc.)
                pcm_data = b''
                while True:
                    chunk_hdr = f.read(8)
                    if len(chunk_hdr) < 8:
                        break
                    chunk_id = chunk_hdr[:4]
                    chunk_size = struct.unpack('<I', chunk_hdr[4:])[0]
                    if chunk_id == b'data':
                        pcm_data = f.read(chunk_size)
                        break
                    f.seek(chunk_size, 1)

            if not pcm_data:
                continue

            # Mix into timeline (additive with clamping)
            offset = start_sample * n_channels * 2
            mix_len = min(len(pcm_data), len(timeline) - offset) - 1
            for j in range(0, mix_len, 2):
                src = struct.unpack_from('<h', pcm_data, j)[0]
                dst = struct.unpack_from('<h', timeline, offset + j)[0]
                mixed = max(-32768, min(32767, src + dst))
                struct.pack_into('<h', timeline, offset + j, mixed)
        except Exception:
            continue

    # Write output WAV
    with open(output_path, "wb") as f:
        # WAV header
        data_size = len(timeline)
        f.write(b"RIFF")
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack('<I', 16))  # chunk size
        f.write(struct.pack('<H', 1))   # PCM
        f.write(struct.pack('<H', n_channels))
        bits_per_sample = 16
        bytes_per_sample = bits_per_sample // 8
        block_align = n_channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', byte_rate))
        f.write(struct.pack('<H', block_align))
        f.write(struct.pack('<H', bits_per_sample))
        f.write(b"data")
        f.write(struct.pack('<I', data_size))
        f.write(timeline)


def _get_duration(path: Path, ffmpeg: str = "ffmpeg") -> float:
    """Get audio/video duration using ffprobe."""
    try:
        ffprobe = ffmpeg.replace("ffmpeg", "ffprobe") if "ffmpeg" in ffmpeg else "ffprobe"
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except Exception:
        return 0
