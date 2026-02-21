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
import math
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
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

        # Step 3: Transcribe speech from audio (narrator's voice only)
        self._report("transcribe", 0.0, "Loading ASR model...")
        self.segments = self._transcribe(audio_raw)
        self._report("transcribe", 1.0, f"Transcribed {len(self.segments)} segments")

        text_segments = [s for s in self.segments if s.get("text", "").strip()]
        if not text_segments:
            raise RuntimeError("No speech detected in the video")

        # Step 4: Translate each segment (preserving timestamps for scene sync)
        self._report("translate", 0.0, "Translating segments to Hindi...")
        self._translate_segments(text_segments)
        self.segments = text_segments
        self._report("translate", 1.0, "Translation complete")

        # Write Hindi SRT (per-segment subtitles with proper timestamps)
        srt_hi = self.cfg.work_dir / "transcript_hi.srt"
        write_srt(self.segments, srt_hi, text_key="text_translated")

        # Step 5+6: Process video in 1-minute chunks for tight scene sync
        video_duration = self._get_duration(video_path)
        chunk_secs = 60.0
        num_chunks = max(1, math.ceil(video_duration / chunk_secs))

        self._report("synthesize", 0.0,
                     f"Processing {num_chunks} minute chunks ({self.cfg.tts_voice})...")

        chunk_outputs = []
        for cidx in range(num_chunks):
            chunk_start = cidx * chunk_secs
            chunk_end = min((cidx + 1) * chunk_secs, video_duration)
            chunk_len = chunk_end - chunk_start
            prefix = f"c{cidx:02d}_"

            pct = cidx / num_chunks
            self._report("synthesize", pct * 0.9,
                         f"Chunk {cidx + 1}/{num_chunks}: splitting video...")

            # 5a. Split video into this minute chunk
            chunk_vid = self.cfg.work_dir / f"{prefix}video.mp4"
            self._split_video(video_path, chunk_start, chunk_len, chunk_vid)

            # 5b. Get segments that belong to this chunk, adjust to chunk-relative time
            chunk_segs = []
            for seg in text_segments:
                if seg["start"] >= chunk_start and seg["start"] < chunk_end:
                    chunk_segs.append({
                        "start": seg["start"] - chunk_start,
                        "end": min(seg["end"], chunk_end) - chunk_start,
                        "text": seg.get("text", ""),
                        "text_translated": seg.get("text_translated", seg.get("text", "")),
                    })

            if not chunk_segs:
                # No speech in this chunk — use silent audio
                silent_wav = self.cfg.work_dir / f"{prefix}silent.wav"
                subprocess.run(
                    [self._ffmpeg, "-y", "-f", "lavfi", "-i",
                     f"anullsrc=r={self.SAMPLE_RATE}:cl=stereo",
                     "-t", f"{chunk_len:.3f}",
                     "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                     str(silent_wav)],
                    check=True, capture_output=True,
                )
                chunk_audio = silent_wav
            else:
                # 5c. TTS + align within this chunk
                self._report("synthesize", pct * 0.9 + 0.02,
                             f"Chunk {cidx + 1}/{num_chunks}: synthesizing {len(chunk_segs)} segments...")
                chunk_audio = self._tts_time_aligned(chunk_segs, chunk_len, prefix=prefix)

            # 5d. Mix original audio if requested
            if self.cfg.mix_original:
                chunk_orig = self.cfg.work_dir / f"{prefix}orig.wav"
                subprocess.run(
                    [self._ffmpeg, "-y", "-ss", f"{chunk_start:.3f}",
                     "-i", str(audio_raw), "-t", f"{chunk_len:.3f}",
                     "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                     str(chunk_orig)],
                    check=True, capture_output=True,
                )
                chunk_audio = self._mix_audio(chunk_orig, chunk_audio, self.cfg.original_volume)

            # 5e. Mux this chunk
            chunk_out = self.cfg.work_dir / f"{prefix}dubbed.mp4"
            self._mux_replace_audio(chunk_vid, chunk_audio, chunk_out)
            chunk_outputs.append(chunk_out)

        self._report("synthesize", 1.0, "All chunks synthesized")

        # Step 6: Concatenate all dubbed chunks into final video
        self._report("assemble", 0.0, f"Assembling {num_chunks} chunks into final video...")
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

        if len(chunk_outputs) == 1:
            shutil.copy2(chunk_outputs[0], self.cfg.output_path)
        else:
            self._concatenate_videos(chunk_outputs, self.cfg.output_path)

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

            # Get video title first (separate call) — try without cookies
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

    # ── Step 3: Transcribe speech from audio ─────────────────────────────
    def _transcribe(self, wav_path: Path) -> List[Dict]:
        """Transcribe speech from audio using Whisper (picks up only spoken words)."""
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

    # ── Step 4: Translate full narrative ─────────────────────────────────
    def _translate_full_narrative(self, text_segments: List[Dict], speech_duration: float = 0) -> tuple:
        """Join all speech into one narrative, translate as a whole."""
        # Combine all transcribed text into one continuous story
        full_text = " ".join(s.get("text", "").strip() for s in text_segments if s.get("text", "").strip())
        self._report("translate", 0.2, f"Translating {len(full_text)} characters...")

        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if gemini_key:
            translated_text = self._translate_with_gemini(full_text, gemini_key, speech_duration)
        else:
            self._report("translate", 0.25, "No GEMINI_API_KEY found, using Google Translate...")
            translated_text = self._translate_with_google(full_text)

        return full_text, translated_text

    def _translate_with_gemini(self, full_text: str, api_key: str, speech_duration: float = 0) -> str:
        """Translate using Gemini LLM for natural, fluent Hindi."""
        from google import genai

        client = genai.Client(api_key=api_key)

        lang_names = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
                      "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
                      "pa": "Punjabi", "ur": "Urdu"}
        target_name = lang_names.get(self.cfg.target_language, self.cfg.target_language)

        # Calculate word count guidance for duration matching
        word_count = len(full_text.split())
        duration_hint = ""
        if speech_duration > 0:
            # Hindi TTS speaks ~130-150 words/min. Target slightly fewer words
            # to avoid needing extreme tempo adjustment (capped at 1.2x)
            target_words = int(speech_duration / 60 * 135)  # ~135 Hindi words/min
            duration_hint = (
                f"IMPORTANT TIMING CONSTRAINT: The original narration is {int(speech_duration)} seconds long "
                f"({word_count} English words). Your {target_name} translation will be spoken by TTS and "
                f"MUST fit within this duration. Aim for approximately {target_words} {target_name} words. "
                f"Be concise — use shorter phrases where possible without losing meaning. "
                f"Avoid filler words and unnecessary elaboration. "
            )

        # Gemini free tier: 10 RPM, so split large texts into chunks
        chunks = self._split_text_for_translation(full_text, max_chars=8000)
        translated_parts = []
        chunk_duration = speech_duration / len(chunks) if speech_duration > 0 else 0

        for i, chunk in enumerate(chunks):
            chunk_words = len(chunk.split())
            chunk_target = int(chunk_duration / 60 * 135) if chunk_duration > 0 else 0
            chunk_hint = ""
            if chunk_target > 0:
                chunk_hint = (
                    f"This chunk has {chunk_words} English words and must fit in ~{int(chunk_duration)} seconds. "
                    f"Aim for ~{chunk_target} {target_name} words. "
                )

            prompt = (
                f"Translate the following English narration into natural, fluent {target_name}. "
                f"This is a voiceover script for a dubbed video, so it must sound like a native "
                f"{target_name} speaker is narrating — conversational, smooth, and natural. "
                f"{duration_hint}{chunk_hint}"
                f"Do NOT translate literally word-by-word. Adapt idioms and phrasing to sound "
                f"natural in {target_name}. Keep proper nouns (names, places, brands) as-is. "
                f"Output ONLY the {target_name} translation, nothing else.\n\n"
                f"{chunk}"
            )

            retries = 3
            for attempt in range(retries):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                    )
                    translated_parts.append(response.text.strip())
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.2 + 0.6 * (i / len(chunks)),
                                     f"Rate limited, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        self._report("translate", 0.2, f"Gemini failed: {e}, falling back to Google Translate...")
                        return self._translate_with_google(full_text)

            self._report("translate", 0.2 + 0.8 * ((i + 1) / len(chunks)),
                         f"Translated chunk {i + 1}/{len(chunks)} (Gemini)")

        return " ".join(translated_parts)

    def _translate_with_google(self, full_text: str) -> str:
        """Fallback: translate using free Google Translate via deep-translator."""
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="auto", target=self.cfg.target_language)
        chunks = self._split_text_for_translation(full_text, max_chars=4500)
        translated_parts = []

        for i, chunk in enumerate(chunks):
            retries = 3
            for attempt in range(retries):
                try:
                    translated_parts.append(translator.translate(chunk))
                    break
                except Exception:
                    if attempt < retries - 1:
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        translated_parts.append(chunk)

            self._report("translate", 0.2 + 0.8 * ((i + 1) / len(chunks)),
                         f"Translated chunk {i + 1}/{len(chunks)}")

        return " ".join(translated_parts)

    @staticmethod
    def _split_text_for_translation(text: str, max_chars: int = 4500) -> List[str]:
        """Split text into chunks at sentence boundaries for translation API limits."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        current = ""
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?।])\s+', text)

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_chars and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = (current + " " + sentence).strip()

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text]

    # ── Step 4b: Segment-level translation ────────────────────────────────
    def _translate_segments(self, segments):
        """Translate each segment individually, preserving timestamps for sync."""
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if gemini_key:
            self._translate_segments_gemini(segments, gemini_key)
        else:
            self._report("translate", 0.1, "No GEMINI_API_KEY, using Google Translate...")
            self._translate_segments_google(segments)

    def _translate_segments_gemini(self, segments, api_key):
        """Translate segments in numbered batches using Gemini for context-aware output."""
        from google import genai
        client = genai.Client(api_key=api_key)

        lang_names = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
                      "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
                      "pa": "Punjabi", "ur": "Urdu"}
        target_name = lang_names.get(self.cfg.target_language, self.cfg.target_language)

        batch_size = 30
        total_batches = (len(segments) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            # Build numbered input with duration hints
            lines = []
            for i, seg in enumerate(batch):
                duration = seg["end"] - seg["start"]
                lines.append(f"{i+1}. [{duration:.1f}s] {seg['text']}")

            prompt = (
                f"Translate each numbered line below from English to natural, fluent {target_name}. "
                f"This is a voiceover script for video dubbing — it must sound like a native "
                f"{target_name} speaker narrating, conversational and smooth. "
                f"The time in brackets [Xs] is how long the original line takes when spoken. "
                f"Keep each translation concise enough to be spoken in roughly that duration. "
                f"Do NOT translate literally; adapt idioms to sound natural in {target_name}. "
                f"Keep proper nouns (names, places, brands) as-is. "
                f"Output ONLY the numbered {target_name} translations, one per line, "
                f"matching the input numbering exactly.\n\n"
                + "\n".join(lines)
            )

            retries = 3
            success = False
            for attempt in range(retries):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash", contents=prompt)
                    translations = self._parse_numbered_translations(response.text, len(batch))
                    for i, seg in enumerate(batch):
                        seg["text_translated"] = translations[i] if translations[i] else seg["text"]
                    success = True
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.1 + 0.8 * (batch_idx / total_batches),
                                     f"Rate limited, retrying in {wait}s...")
                        time.sleep(wait)

            if not success:
                self._report("translate", 0.1, "Gemini failed, using Google Translate for batch...")
                for seg in batch:
                    seg["text_translated"] = self._translate_single_google(seg["text"])

            self._report("translate", 0.1 + 0.9 * ((batch_idx + 1) / total_batches),
                         f"Translated batch {batch_idx + 1}/{total_batches}")

    def _translate_segments_google(self, segments):
        """Fallback: translate each segment using Google Translate."""
        for i, seg in enumerate(segments):
            seg["text_translated"] = self._translate_single_google(seg["text"])
            self._report("translate", 0.1 + 0.9 * ((i + 1) / len(segments)),
                         f"Translated {i + 1}/{len(segments)} segments")

    def _translate_single_google(self, text: str) -> str:
        """Translate a single text with Google Translate."""
        from deep_translator import GoogleTranslator
        try:
            translator = GoogleTranslator(source="auto", target=self.cfg.target_language)
            return translator.translate(text) or text
        except Exception:
            return text

    @staticmethod
    def _parse_numbered_translations(text: str, expected_count: int) -> List[str]:
        """Parse numbered translation output from Gemini."""
        lines = text.strip().split("\n")
        translations = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Match: "1. translation" or "1) translation" or "1. [3.2s] translation"
            match = re.match(r'\s*\d+[\.\)]\s*(?:\[[\d\.]+s?\]\s*)?(.*)', line)
            if match:
                trans = match.group(1).strip()
                if trans:
                    translations.append(trans)
        # Pad with empty strings if Gemini returned fewer lines
        while len(translations) < expected_count:
            translations.append("")
        return translations[:expected_count]

    # ── Step 5: Continuous TTS ────────────────────────────────────────────
    def _tts_continuous(self, translated_text: str) -> Path:
        """Synthesize the entire translated narrative as ONE single TTS call."""
        import edge_tts

        out_mp3 = self.cfg.work_dir / "tts_full.mp3"
        out_wav = self.cfg.work_dir / "tts_full.wav"
        voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        self._report("synthesize", 0.1, "Generating speech (single voice)...")

        async def synthesize():
            communicate = edge_tts.Communicate(translated_text, voice, rate=rate)
            await communicate.save(str(out_mp3))

        asyncio.run(synthesize())

        if not out_mp3.exists() or out_mp3.stat().st_size == 0:
            raise RuntimeError("TTS synthesis produced no audio")

        self._report("synthesize", 0.8, "Converting to WAV...")

        # Convert to WAV
        subprocess.run(
            [
                self._ffmpeg, "-y", "-i", str(out_mp3),
                "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                str(out_wav),
            ],
            check=True, capture_output=True,
        )
        out_mp3.unlink(missing_ok=True)

        return out_wav

    # ── Step 5b: Time-aligned TTS ─────────────────────────────────────────
    def _tts_time_aligned(self, segments, total_duration, prefix="", progress_base=0.0, progress_span=1.0):
        """Generate TTS per segment, adjust tempo to fit time slot, place at original timestamps.

        prefix: unique prefix for temp file names (e.g. "c01_" for chunk 1)
        progress_base/span: for reporting progress within a larger job
        """
        import edge_tts
        voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        # Generate all TTS mp3 files in a single async event loop
        async def generate_all():
            for i, seg in enumerate(segments):
                text = seg.get("text_translated", seg["text"]).strip()
                if not text:
                    continue
                mp3 = self.cfg.work_dir / f"{prefix}seg_{i:04d}.mp3"
                try:
                    comm = edge_tts.Communicate(text, voice, rate=rate)
                    await comm.save(str(mp3))
                    seg["_tts_mp3"] = mp3
                except Exception:
                    pass  # skip failed segments

        asyncio.run(generate_all())

        # Convert to WAV and adjust tempo per segment to fit its time slot
        tts_data = []
        for i, seg in enumerate(segments):
            mp3 = seg.pop("_tts_mp3", None)
            if not mp3 or not mp3.exists():
                continue

            wav = self.cfg.work_dir / f"{prefix}seg_{i:04d}.wav"
            subprocess.run(
                [self._ffmpeg, "-y", "-i", str(mp3),
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 str(wav)],
                check=True, capture_output=True,
            )
            mp3.unlink(missing_ok=True)

            # Available time = gap from this segment's start to the next segment's start
            seg_duration = seg["end"] - seg["start"]
            if i < len(segments) - 1:
                available = segments[i + 1]["start"] - seg["start"]
            else:
                available = (total_duration - seg["start"]) if total_duration > 0 else seg_duration
            available = max(available, seg_duration)  # at least the segment's own duration

            tts_dur = self._get_duration(wav)

            # Speed up TTS if it doesn't fit the time slot (up to 1.8x for segments)
            if tts_dur > 0 and available > 0 and tts_dur > available * 1.05:
                ratio = tts_dur / available
                ratio = min(ratio, 1.8)  # cap at 1.8x per segment
                if ratio > 1.05:
                    adjusted = self.cfg.work_dir / f"{prefix}seg_{i:04d}_adj.wav"
                    tempo = ratio
                    filters = []
                    while tempo > 2.0:
                        filters.append("atempo=2.0")
                        tempo /= 2.0
                    filters.append(f"atempo={tempo:.4f}")
                    subprocess.run(
                        [self._ffmpeg, "-y", "-i", str(wav),
                         "-filter:a", ",".join(filters),
                         "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                         str(adjusted)],
                        check=True, capture_output=True,
                    )
                    wav = adjusted
                    tts_dur = self._get_duration(wav)

            tts_data.append({
                "start": seg["start"],
                "wav": wav,
                "duration": tts_dur,
            })

        return self._build_timeline(tts_data, total_duration, prefix)

    def _build_timeline(self, tts_data, total_duration, prefix=""):
        """Place TTS segments at their original timestamps on a silent audio track."""
        total_samples = int((total_duration + 0.5) * self.SAMPLE_RATE)
        bytes_per_frame = 2 * self.N_CHANNELS  # 16-bit stereo = 4 bytes
        timeline = bytearray(total_samples * bytes_per_frame)

        for seg in tts_data:
            start_byte = int(seg["start"] * self.SAMPLE_RATE) * bytes_per_frame

            with wave.open(str(seg["wav"]), 'rb') as w:
                raw = w.readframes(w.getnframes())

            end_byte = min(start_byte + len(raw), len(timeline))
            copy_len = end_byte - start_byte
            if copy_len > 0:
                timeline[start_byte:end_byte] = raw[:copy_len]

        output = self.cfg.work_dir / f"{prefix}tts_aligned.wav"
        with wave.open(str(output), 'wb') as w:
            w.setnchannels(self.N_CHANNELS)
            w.setsampwidth(2)
            w.setframerate(self.SAMPLE_RATE)
            w.writeframes(bytes(timeline))

        return output

    # ── Duration & tempo adjustment ───────────────────────────────────────
    def _get_duration(self, media_path: Path) -> float:
        """Get duration of a media file in seconds using ffprobe."""
        ffprobe = str(Path(self._ffmpeg).parent / "ffprobe")
        if sys.platform == "win32" and not ffprobe.endswith(".exe"):
            ffprobe += ".exe"
        try:
            result = subprocess.run(
                [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(media_path)],
                capture_output=True, text=True, timeout=15,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _adjust_tempo(self, wav_path: Path, ratio: float) -> Path:
        """Speed up or slow down audio to match video duration.

        ratio = tts_duration / video_duration
        ratio > 1 means TTS is longer → speed up (atempo > 1)
        ratio < 1 means TTS is shorter → slow down (atempo < 1)
        """
        adjusted = self.cfg.work_dir / "tts_adjusted.wav"

        # ffmpeg atempo filter accepts 0.5 to 100.0
        # For values outside 0.5-2.0, chain multiple filters
        tempo = ratio
        filters = []
        while tempo > 2.0:
            filters.append("atempo=2.0")
            tempo /= 2.0
        while tempo < 0.5:
            filters.append("atempo=0.5")
            tempo /= 0.5
        filters.append(f"atempo={tempo:.4f}")

        filter_str = ",".join(filters)
        subprocess.run(
            [
                self._ffmpeg, "-y", "-i", str(wav_path),
                "-filter:a", filter_str,
                "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                str(adjusted),
            ],
            check=True, capture_output=True,
        )
        return adjusted

    def _adjust_video_duration(self, video_path: Path, target_duration: float) -> Path:
        """Adjust video duration to match the dubbed audio using setpts filter.

        If dubbed audio is longer than video → slow down video (scenes last longer).
        If dubbed audio is shorter than video → speed up video (scenes go faster).
        """
        video_duration = self._get_duration(video_path)
        if video_duration <= 0 or target_duration <= 0:
            return video_path

        # PTS factor: >1 slows video down, <1 speeds it up
        pts_factor = target_duration / video_duration
        if abs(pts_factor - 1.0) < 0.02:  # Less than 2% difference, skip
            return video_path

        adjusted = self.cfg.work_dir / "video_adjusted.mp4"
        self._report("assemble", 0.1,
                     f"Adjusting video speed ({1/pts_factor:.2f}x) to match audio...")

        # setpts=PTS*factor changes video timing
        # factor > 1 → slower (stretches video), factor < 1 → faster (compresses video)
        # fps filter re-establishes constant frame rate after pts change
        subprocess.run(
            [
                self._ffmpeg, "-y", "-i", str(video_path),
                "-filter:v", f"setpts={pts_factor:.6f}*PTS",
                "-an",  # Drop original audio (we'll add dubbed audio)
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                str(adjusted),
            ],
            check=True, capture_output=True,
        )
        return adjusted

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

    # ── Video split / concat ─────────────────────────────────────────────
    def _split_video(self, video_path: Path, start: float, duration: float, output_path: Path):
        """Extract a clip from the video using stream copy (fast, no re-encode)."""
        subprocess.run(
            [self._ffmpeg, "-y",
             "-ss", f"{start:.3f}", "-i", str(video_path),
             "-t", f"{duration:.3f}",
             "-c", "copy", "-an",  # copy video only, drop audio
             str(output_path)],
            check=True, capture_output=True,
        )

    def _concatenate_videos(self, video_paths: List[Path], output_path: Path):
        """Concatenate multiple video files using ffmpeg concat demuxer."""
        concat_list = self.cfg.work_dir / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for vp in video_paths:
                # ffmpeg concat needs forward slashes even on Windows
                safe_path = str(vp).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")
        subprocess.run(
            [self._ffmpeg, "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list), "-c", "copy",
             str(output_path)],
            check=True, capture_output=True,
        )

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
