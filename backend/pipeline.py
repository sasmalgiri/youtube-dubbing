"""
Dubbing Pipeline with Translation Support
==========================================
Refactored from pipeline_v2 with:
- Callback-based progress reporting (for SSE)
- Translation step (deep-translator)
- Hindi TTS by default
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from srt_utils import write_srt


# ── Types ────────────────────────────────────────────────────────────────────
ProgressCallback = Callable[[str, float, str], None]

STEPS = ["download", "extract", "transcribe", "translate", "synthesize", "assemble"]
STEP_WEIGHTS = {
    "download": 0.15,
    "extract": 0.05,
    "transcribe": 0.25,
    "translate": 0.15,
    "synthesize": 0.30,
    "assemble": 0.10,
}


@dataclass
class PipelineConfig:
    source: str
    work_dir: Path
    output_path: Path
    target_language: str = "hi"
    asr_model: str = "small"
    tts_voice: str = "hi-IN-SwaraNeural"
    tts_rate: str = "+0%"
    mix_original: bool = True
    original_volume: float = 0.15
    time_aligned: bool = True


class Pipeline:
    """Dubbing pipeline with translation and callback-based progress."""

    SAMPLE_RATE = 48000
    N_CHANNELS = 2

    def __init__(self, cfg: PipelineConfig, on_progress: Optional[ProgressCallback] = None):
        self.cfg = cfg
        self._on_progress = on_progress or (lambda *_: None)
        self.segments: List[Dict] = []
        self.video_title: str = ""
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)

        # Resolve executable paths from the venv
        venv_scripts = Path(sys.executable).parent
        self._ytdlp = str(venv_scripts / "yt-dlp.exe") if sys.platform == "win32" else "yt-dlp"
        if not Path(self._ytdlp).exists():
            self._ytdlp = shutil.which("yt-dlp") or "yt-dlp"
        self._ffmpeg = "ffmpeg"  # resolved in _ensure_ffmpeg

    def _report(self, step: str, progress: float, message: str):
        """Report progress to callback."""
        self._on_progress(step, min(progress, 1.0), message)

    # ── Main entry ───────────────────────────────────────────────────────
    def run(self):
        """Execute the full dubbing pipeline."""
        self._ensure_ffmpeg()

        # Step 1: Download / ingest
        self._report("download", 0.0, "Downloading video...")
        video_path = self._ingest_source(self.cfg.source)
        self._report("download", 1.0, f"Downloaded: {video_path.name}")

        # Step 2: Extract audio
        self._report("extract", 0.0, "Extracting audio...")
        audio_raw = self._extract_audio(video_path)
        self._report("extract", 1.0, "Audio extracted")

        # Step 3: Transcribe
        self._report("transcribe", 0.0, "Loading ASR model...")
        self.segments = self._transcribe(audio_raw)
        self._report("transcribe", 1.0, f"Transcribed {len(self.segments)} segments")

        # Write English SRT
        srt_en = self.cfg.work_dir / "transcript_en.srt"
        write_srt(self.segments, srt_en, text_key="text")

        # Check for speech
        text_segments = [s for s in self.segments if s.get("text", "").strip()]
        if not text_segments:
            raise RuntimeError("No speech detected in the video")

        # Step 4: Translate
        self._report("translate", 0.0, "Translating to Hindi...")
        self.segments = self._translate_segments(self.segments)
        self._report("translate", 1.0, f"Translated {len(text_segments)} segments")

        # Write Hindi SRT
        srt_hi = self.cfg.work_dir / "transcript_hi.srt"
        write_srt(self.segments, srt_hi, text_key="text_translated")

        # Step 5: Synthesize TTS
        self._report("synthesize", 0.0, f"Synthesizing Hindi audio ({self.cfg.tts_voice})...")
        if self.cfg.time_aligned:
            tts_wav = self._tts_time_aligned(text_segments)
        else:
            tts_wav = self._tts_concatenated(text_segments)

        if not tts_wav.exists():
            raise RuntimeError("TTS synthesis failed - no output file")
        self._report("synthesize", 1.0, "Audio synthesized")

        # Step 6: Assemble
        self._report("assemble", 0.0, "Assembling final video...")

        final_audio = tts_wav
        if self.cfg.mix_original:
            self._report("assemble", 0.3, f"Mixing original audio ({self.cfg.original_volume:.0%})...")
            final_audio = self._mix_audio(audio_raw, tts_wav, self.cfg.original_volume)

        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._mux_replace_audio(video_path, final_audio, self.cfg.output_path)

        # Copy SRT to output
        out_srt = self.cfg.output_path.parent / "subtitles_hi.srt"
        shutil.copy2(srt_hi, out_srt)

        self._report("assemble", 1.0, "Done!")

    # ── FFmpeg check ─────────────────────────────────────────────────────
    def _ensure_ffmpeg(self):
        if shutil.which("ffmpeg") is None and sys.platform == "win32":
            # Try WinGet install path
            localappdata = os.environ.get("LOCALAPPDATA", "")
            winget_ffmpeg = Path(localappdata) / "Microsoft" / "WinGet" / "Packages"
            if winget_ffmpeg.exists():
                for pkg in winget_ffmpeg.iterdir():
                    if pkg.name.startswith("Gyan.FFmpeg"):
                        for sub in pkg.rglob("ffmpeg.exe"):
                            bin_path = sub.parent
                            os.environ["PATH"] = str(bin_path) + os.pathsep + os.environ.get("PATH", "")
                            break

            # Also try common install paths
            common_paths = [
                Path(os.environ.get("ProgramFiles", "")) / "ffmpeg" / "bin",
                Path(os.environ.get("ProgramFiles(x86)", "")) / "ffmpeg" / "bin",
                Path(os.environ.get("USERPROFILE", "")) / "scoop" / "shims",
            ]
            for p in common_paths:
                if (p / "ffmpeg.exe").exists():
                    os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
                    break

            # Try refreshing PATH from system environment
            if shutil.which("ffmpeg") is None:
                sys_path = os.popen('powershell.exe -NoProfile -Command "[System.Environment]::GetEnvironmentVariable(\'Path\',\'Machine\')"').read().strip()
                user_path = os.popen('powershell.exe -NoProfile -Command "[System.Environment]::GetEnvironmentVariable(\'Path\',\'User\')"').read().strip()
                if sys_path or user_path:
                    os.environ["PATH"] = sys_path + os.pathsep + user_path + os.pathsep + os.environ.get("PATH", "")

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "FFmpeg not found! Install: winget install Gyan.FFmpeg"
            )
        self._ffmpeg = ffmpeg_path

    # ── Step 1: Ingest ───────────────────────────────────────────────────
    def _ingest_source(self, src: str) -> Path:
        if re.match(r"^https?://", src):
            out_tpl = str(self.cfg.work_dir / "source.%(ext)s")

            # Get video title first (separate call)
            try:
                title_result = subprocess.run(
                    [self._ytdlp, "--print", "%(title)s", "--no-download", src],
                    capture_output=True, text=True, timeout=30,
                )
                if title_result.returncode == 0 and title_result.stdout.strip():
                    self.video_title = title_result.stdout.strip().split("\n")[0]
            except Exception:
                self.video_title = "Untitled"

            self._report("download", 0.2, f"Downloading: {self.video_title}")

            # Download video
            try:
                subprocess.run(
                    [
                        self._ytdlp,
                        "--ffmpeg-location", self._ffmpeg,
                        "-f", "bv*+ba/b",
                        "--merge-output-format", "mp4",
                        "-o", out_tpl,
                        src,
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "Unknown error")
                raise RuntimeError(
                    f"yt-dlp failed: {stderr}\n"
                    "Ensure you own the content or try: yt-dlp --cookies-from-browser chrome <url>"
                ) from e

            # Find downloaded file
            mp4 = list(self.cfg.work_dir.glob("source.mp4"))
            if mp4:
                return mp4[0]
            all_sources = list(self.cfg.work_dir.glob("source.*"))
            if all_sources:
                return all_sources[0]
            raise RuntimeError("Download completed but no video file found in work directory")

        p = Path(src)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"Source not found: {src}")
        self.video_title = p.stem
        return p

    # ── Step 2: Extract audio ────────────────────────────────────────────
    def _extract_audio(self, video_path: Path) -> Path:
        wav = self.cfg.work_dir / "audio_raw.wav"
        subprocess.run(
            [
                self._ffmpeg, "-y", "-i", str(video_path),
                "-vn", "-ac", str(self.N_CHANNELS), "-ar", str(self.SAMPLE_RATE),
                "-acodec", "pcm_s16le", str(wav),
            ],
            check=True,
            capture_output=True,
        )
        return wav

    # ── Step 3: Transcribe ───────────────────────────────────────────────
    def _transcribe(self, wav_path: Path) -> List[Dict]:
        from faster_whisper import WhisperModel

        self._report("transcribe", 0.1, f"Loading model ({self.cfg.asr_model})...")
        model = WhisperModel(self.cfg.asr_model, device="cpu", compute_type="int8")

        self._report("transcribe", 0.2, "Transcribing audio...")
        seg_iter, info = model.transcribe(str(wav_path), vad_filter=True)

        segments: List[Dict] = []
        for seg in seg_iter:
            segments.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            })
            self._report(
                "transcribe",
                min(0.2 + 0.8 * (len(segments) / max(len(segments) + 5, 1)), 0.95),
                f"Transcribed {len(segments)} segments...",
            )

        return segments

    # ── Step 4: Translate ────────────────────────────────────────────────
    def _translate_segments(self, segments: List[Dict]) -> List[Dict]:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="auto", target=self.cfg.target_language)
        text_segments = [s for s in segments if s.get("text", "").strip()]
        total = len(text_segments)

        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:
                seg["text_translated"] = ""
                continue

            retries = 3
            for attempt in range(retries):
                try:
                    seg["text_translated"] = translator.translate(text)
                    break
                except Exception:
                    if attempt < retries - 1:
                        time.sleep(1 * (attempt + 1))
                    else:
                        seg["text_translated"] = text  # fallback to original

            self._report(
                "translate",
                (i + 1) / max(total, 1),
                f"Translated {i + 1}/{total} segments",
            )

        return segments

    # ── Step 5a: TTS time-aligned ────────────────────────────────────────
    def _tts_time_aligned(self, text_segments: List[Dict]) -> Path:
        import edge_tts

        out_wav = self.cfg.work_dir / "tts_aligned.wav"
        voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        async def synthesize_segment(text: str, seg_out: Path):
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(str(seg_out))

        async def run_all():
            seg_data: List[tuple] = []
            total = len(text_segments)

            for i, seg in enumerate(text_segments):
                text = seg.get("text_translated", seg.get("text", "")).strip()
                if not text:
                    continue

                start_time = seg.get("start", 0.0)
                seg_mp3 = self.cfg.work_dir / f"tts_seg_{i:04d}.mp3"
                seg_wav = self.cfg.work_dir / f"tts_seg_{i:04d}.wav"

                try:
                    await synthesize_segment(text, seg_mp3)
                    subprocess.run(
                        [
                            self._ffmpeg, "-y", "-i", str(seg_mp3),
                            "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                            str(seg_wav),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    seg_data.append((start_time, seg_wav))
                except Exception:
                    pass  # skip failed segments
                finally:
                    seg_mp3.unlink(missing_ok=True)

                self._report("synthesize", (i + 1) / total, f"Synthesized {i + 1}/{total} segments")

            self._build_aligned_wav(seg_data, out_wav)

            for _, wav_path in seg_data:
                wav_path.unlink(missing_ok=True)

        asyncio.run(run_all())
        return out_wav

    # ── Step 5b: TTS concatenated ────────────────────────────────────────
    def _tts_concatenated(self, text_segments: List[Dict]) -> Path:
        import edge_tts

        out_wav = self.cfg.work_dir / "tts_concat.wav"
        voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        async def synthesize_segment(text: str, seg_out: Path):
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(str(seg_out))

        async def run_all():
            seg_wavs: List[Path] = []
            total = len(text_segments)

            for i, seg in enumerate(text_segments):
                text = seg.get("text_translated", seg.get("text", "")).strip()
                if not text:
                    continue

                seg_mp3 = self.cfg.work_dir / f"tts_seg_{i:04d}.mp3"
                seg_wav = self.cfg.work_dir / f"tts_seg_{i:04d}.wav"

                try:
                    await synthesize_segment(text, seg_mp3)
                    subprocess.run(
                        [
                            self._ffmpeg, "-y", "-i", str(seg_mp3),
                            "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                            str(seg_wav),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    seg_wavs.append(seg_wav)
                except Exception:
                    pass
                finally:
                    seg_mp3.unlink(missing_ok=True)

                self._report("synthesize", (i + 1) / total, f"Synthesized {i + 1}/{total} segments")

            if seg_wavs:
                with wave.open(str(out_wav), "wb") as wf:
                    wf.setnchannels(self.N_CHANNELS)
                    wf.setsampwidth(2)
                    wf.setframerate(self.SAMPLE_RATE)
                    for sw in seg_wavs:
                        with wave.open(str(sw), "rb") as rf:
                            wf.writeframes(rf.readframes(rf.getnframes()))
                        sw.unlink(missing_ok=True)

        asyncio.run(run_all())
        return out_wav

    # ── WAV alignment ────────────────────────────────────────────────────
    def _build_aligned_wav(self, seg_data: List[tuple], out_wav: Path):
        if not seg_data:
            return

        max_end = 0.0
        for start_time, wav_path in seg_data:
            with wave.open(str(wav_path), "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
                max_end = max(max_end, start_time + duration)

        total_samples = int(max_end * self.SAMPLE_RATE) + self.SAMPLE_RATE
        audio_buffer = [0] * (total_samples * self.N_CHANNELS)

        for start_time, wav_path in seg_data:
            start_sample = int(start_time * self.SAMPLE_RATE)
            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                samples = struct.unpack(f"<{len(frames) // 2}h", frames)

                for j, sample in enumerate(samples):
                    idx = start_sample * self.N_CHANNELS + j
                    if idx < len(audio_buffer):
                        audio_buffer[idx] = max(-32768, min(32767, audio_buffer[idx] + sample))

        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(self.N_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(struct.pack(f"<{len(audio_buffer)}h", *audio_buffer))

    # ── Audio mixing ─────────────────────────────────────────────────────
    def _mix_audio(self, original: Path, tts: Path, original_vol: float) -> Path:
        mixed = self.cfg.work_dir / "audio_mixed.wav"
        subprocess.run(
            [
                self._ffmpeg, "-y",
                "-i", str(tts),
                "-i", str(original),
                "-filter_complex",
                f"[1:a]volume={original_vol}[orig];[0:a][orig]amix=inputs=2:duration=first:dropout_transition=2[out]",
                "-map", "[out]",
                "-ar", str(self.SAMPLE_RATE),
                "-ac", str(self.N_CHANNELS),
                str(mixed),
            ],
            check=True,
            capture_output=True,
        )
        return mixed

    # ── Video muxing ─────────────────────────────────────────────────────
    def _mux_replace_audio(self, video_path: Path, audio_path: Path, output_path: Path):
        subprocess.run(
            [
                self._ffmpeg, "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )


async def list_voices(language_filter: str = "hi"):
    """List available edge-tts voices filtered by language."""
    import edge_tts

    voices = await edge_tts.list_voices()
    if language_filter:
        voices = [v for v in voices if v.get("Locale", "").startswith(language_filter)]
    return voices
