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
    tts_rate: str = "-5%"
    mix_original: bool = False
    original_volume: float = 0.10
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

        # Resolve executable paths
        self._ytdlp = self._find_executable("yt-dlp")
        self._ffmpeg = "ffmpeg"  # resolved in _ensure_ffmpeg

    @staticmethod
    def _find_executable(name: str) -> str:
        """Find an executable by checking venv, PATH, WinGet packages, and system PATH."""
        ext = ".exe" if sys.platform == "win32" else ""
        full_name = name + ext

        # 1. Check venv Scripts dir (where python.exe lives)
        venv_path = Path(sys.executable).parent / full_name
        if venv_path.exists():
            return str(venv_path)

        # 2. Check current PATH
        found = shutil.which(name)
        if found:
            return found

        if sys.platform == "win32":
            # 3. Scan WinGet packages directory
            localappdata = os.environ.get("LOCALAPPDATA", "")
            if localappdata:
                winget_pkgs = Path(localappdata) / "Microsoft" / "WinGet" / "Packages"
                if winget_pkgs.exists():
                    for exe in winget_pkgs.rglob(full_name):
                        os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
                        return str(exe)

            # 4. Refresh PATH from system registry and try again
            try:
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command",
                     "[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    os.environ["PATH"] = result.stdout.strip() + os.pathsep + os.environ.get("PATH", "")
                    found = shutil.which(name)
                    if found:
                        return found
            except Exception:
                pass

        return name  # fallback to bare name

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
        resolved = self._find_executable("ffmpeg")

        # Also scan WinGet install paths as last resort
        if resolved == "ffmpeg" and sys.platform == "win32":
            localappdata = os.environ.get("LOCALAPPDATA", "")
            winget_ffmpeg = Path(localappdata) / "Microsoft" / "WinGet" / "Packages"
            if winget_ffmpeg.exists():
                for exe in winget_ffmpeg.rglob("ffmpeg.exe"):
                    resolved = str(exe)
                    os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
                    break

        if resolved == "ffmpeg" and shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "FFmpeg not found! Install: winget install Gyan.FFmpeg"
            )
        self._ffmpeg = resolved

    # ── Step 1: Ingest ───────────────────────────────────────────────────
    def _ingest_source(self, src: str) -> Path:
        if re.match(r"^https?://", src):
            out_tpl = str(self.cfg.work_dir / "source.%(ext)s")

            # Get video title first (separate call)
            try:
                title_result = subprocess.run(
                    [self._ytdlp, "--cookies-from-browser", "chrome",
                     "--print", "%(title)s", "--no-download", src],
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
                        "--cookies-from-browser", "chrome",
                        "--ffmpeg-location", str(Path(self._ffmpeg).parent),
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
                raise RuntimeError(f"yt-dlp failed: {stderr}") from e

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

    # ── Paragraph grouping ──────────────────────────────────────────────
    @staticmethod
    def _group_into_paragraphs(segments: List[Dict], pause_threshold: float = 1.5) -> List[List[Dict]]:
        """Group nearby segments into paragraphs for natural TTS.

        Segments separated by less than pause_threshold seconds are merged
        into one paragraph. This gives TTS enough context for natural prosody
        instead of synthesizing isolated phrases.
        """
        if not segments:
            return []

        paragraphs: List[List[Dict]] = [[segments[0]]]
        for seg in segments[1:]:
            prev_end = paragraphs[-1][-1].get("end", 0.0)
            cur_start = seg.get("start", 0.0)
            if (cur_start - prev_end) > pause_threshold:
                paragraphs.append([seg])
            else:
                paragraphs[-1].append(seg)

        return paragraphs

    # ── Step 4: Translate ────────────────────────────────────────────────
    def _translate_segments(self, segments: List[Dict]) -> List[Dict]:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="auto", target=self.cfg.target_language)

        # Group into paragraphs for better translation context
        text_segments = [s for s in segments if s.get("text", "").strip()]
        paragraphs = self._group_into_paragraphs(text_segments)
        total_paras = len(paragraphs)

        para_idx = 0
        for para in paragraphs:
            # Join paragraph text for context-aware translation
            full_text = " ".join(s.get("text", "").strip() for s in para)

            retries = 3
            translated = full_text
            for attempt in range(retries):
                try:
                    translated = translator.translate(full_text)
                    break
                except Exception:
                    if attempt < retries - 1:
                        time.sleep(1 * (attempt + 1))

            # Store translated paragraph text on the first segment of the group
            # and mark the rest as part of the group
            for i, seg in enumerate(para):
                if i == 0:
                    seg["text_translated"] = translated
                    seg["_para_start"] = True
                else:
                    seg["text_translated"] = ""
                    seg["_para_start"] = False

            para_idx += 1
            self._report(
                "translate",
                para_idx / max(total_paras, 1),
                f"Translated {para_idx}/{total_paras} paragraphs",
            )

        # Mark empty segments
        for seg in segments:
            if "text_translated" not in seg:
                seg["text_translated"] = ""
                seg["_para_start"] = False

        return segments

    # ── Step 5a: TTS time-aligned ────────────────────────────────────────
    def _tts_time_aligned(self, text_segments: List[Dict]) -> Path:
        import edge_tts

        out_wav = self.cfg.work_dir / "tts_aligned.wav"
        voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        async def synthesize_paragraph(text: str, seg_out: Path):
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(str(seg_out))

        async def run_all():
            seg_data: List[tuple] = []

            # Collect paragraphs (first segment of each group has the text)
            para_segments = [s for s in text_segments
                            if s.get("_para_start", True) and s.get("text_translated", "").strip()]
            total = len(para_segments)

            for i, seg in enumerate(para_segments):
                text = seg["text_translated"].strip()
                start_time = seg.get("start", 0.0)
                seg_mp3 = self.cfg.work_dir / f"tts_para_{i:04d}.mp3"
                seg_wav = self.cfg.work_dir / f"tts_para_{i:04d}.wav"

                try:
                    await synthesize_paragraph(text, seg_mp3)
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
                    pass  # skip failed paragraphs
                finally:
                    seg_mp3.unlink(missing_ok=True)

                self._report("synthesize", (i + 1) / total, f"Synthesized {i + 1}/{total} paragraphs")

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

        async def synthesize_paragraph(text: str, seg_out: Path):
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(str(seg_out))

        async def run_all():
            seg_wavs: List[Path] = []

            para_segments = [s for s in text_segments
                            if s.get("_para_start", True) and s.get("text_translated", "").strip()]
            total = len(para_segments)

            for i, seg in enumerate(para_segments):
                text = seg["text_translated"].strip()
                seg_mp3 = self.cfg.work_dir / f"tts_para_{i:04d}.mp3"
                seg_wav = self.cfg.work_dir / f"tts_para_{i:04d}.wav"

                try:
                    await synthesize_paragraph(text, seg_mp3)
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

                self._report("synthesize", (i + 1) / total, f"Synthesized {i + 1}/{total} paragraphs")

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
