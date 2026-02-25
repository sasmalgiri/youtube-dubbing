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


LANGUAGE_NAMES = {
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "ur": "Urdu", "en": "English", "es": "Spanish",
    "fr": "French", "de": "German", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
    "it": "Italian", "tr": "Turkish",
}

DEFAULT_VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "it": "it-IT-ElsaNeural",
    "tr": "tr-TR-EmelNeural",
    "bn": "bn-IN-TanishaaNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "pa": "pa-IN-GurpreetNeural",
    "ur": "ur-PK-UzmaNeural",
}


@dataclass
class PipelineConfig:
    source: str
    work_dir: Path
    output_path: Path
    source_language: str = "auto"
    target_language: str = "hi"
    asr_model: str = "small"
    tts_voice: str = "hi-IN-SwaraNeural"
    tts_rate: str = "+0%"
    mix_original: bool = False
    original_volume: float = 0.10
    use_chatterbox: bool = True
    use_elevenlabs: bool = False
    use_edge_tts: bool = False


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
        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        self._report("translate", 0.0, f"Translating segments to {target_name}...")
        self._translate_segments(text_segments)
        self.segments = text_segments
        self._report("translate", 1.0, "Translation complete")

        # Write translated SRT (per-segment subtitles with proper timestamps)
        srt_translated = self.cfg.work_dir / f"transcript_{self.cfg.target_language}.srt"
        write_srt(self.segments, srt_translated, text_key="text_translated")

        # Step 5: Generate TTS at natural speed (no speed manipulation)
        self._report("synthesize", 0.0,
                     f"Generating natural speech ({self.cfg.tts_voice})...")
        tts_data = self._generate_tts_natural(text_segments)
        self._report("synthesize", 1.0,
                     f"Generated {len(tts_data)} speech segments")

        # Step 6: Sync video to match natural audio
        # Instead of changing audio speed, adjust video speed per-segment:
        #   TTS longer than original → video slows (scene repeats slightly)
        #   TTS shorter → video speeds up slightly
        #   Gaps between speech play at normal speed
        self._report("assemble", 0.0, "Syncing video to natural speech...")
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        video_duration = self._get_duration(video_path)
        self._build_video_synced(
            video_path, audio_raw, tts_data, video_duration)

        # Copy SRT to output
        out_srt = self.cfg.output_path.parent / f"subtitles_{self.cfg.target_language}.srt"
        shutil.copy2(srt_translated, out_srt)

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
    def _find_cookies_file(self) -> Optional[str]:
        """Find a YouTube cookies file if available."""
        # Check common locations
        for path in [
            Path(__file__).resolve().parent / "cookies.txt",
            Path.home() / "cookies.txt",
            Path("/content/cookies.txt"),  # Colab
        ]:
            if path.exists():
                return str(path)
        return None

    def _ingest_source(self, src: str) -> Path:
        if re.match(r"^https?://", src):
            out_tpl = str(self.cfg.work_dir / "source.%(ext)s")

            # Check for cookies file (needed on Colab/servers)
            cookies_file = self._find_cookies_file()
            cookies_args = ["--cookies", cookies_file] if cookies_file else []

            # Get video title first
            try:
                title_cmd = [self._ytdlp, "--print", "%(title)s", "--no-download"] + cookies_args + [src]
                title_result = subprocess.run(
                    title_cmd, capture_output=True, text=True, timeout=30,
                )
                if title_result.returncode == 0 and title_result.stdout.strip():
                    self.video_title = title_result.stdout.strip().split("\n")[0]
            except Exception:
                self.video_title = "Untitled"

            self._report("download", 0.2, f"Downloading: {self.video_title}")

            # Download video
            try:
                dl_cmd = [
                    self._ytdlp,
                    "--ffmpeg-location", str(Path(self._ffmpeg).parent),
                    "-f", "bv*+ba/b",
                    "--merge-output-format", "mp4",
                    "-o", out_tpl,
                ] + cookies_args + [src]
                subprocess.run(dl_cmd, check=True, capture_output=True)
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

        # Auto-detect GPU: use CUDA if available, else fall back to CPU
        device, compute = "cpu", "int8"
        try:
            import torch
            if torch.cuda.is_available():
                device, compute = "cuda", "float16"
        except ImportError:
            pass

        self._report("transcribe", 0.1, f"Loading model ({self.cfg.asr_model}) on {device.upper()}...")
        model = WhisperModel(self.cfg.asr_model, device=device, compute_type=compute)

        self._report("transcribe", 0.2, "Transcribing audio with word timestamps...")
        # Pass language hint if source language is specified (not auto)
        transcribe_kwargs = {"vad_filter": True, "word_timestamps": True}
        if self.cfg.source_language and self.cfg.source_language != "auto":
            transcribe_kwargs["language"] = self.cfg.source_language
            self._report("transcribe", 0.2, f"Transcribing ({self.cfg.source_language.upper()}) with word timestamps...")
        seg_iter, info = model.transcribe(str(wav_path), **transcribe_kwargs)

        segments: List[Dict] = []
        for seg in seg_iter:
            entry = {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            }
            # Word-level timestamps for fine-grained alignment
            if hasattr(seg, "words") and seg.words:
                entry["words"] = [
                    {"word": w.word.strip(), "start": float(w.start), "end": float(w.end)}
                    for w in seg.words
                ]
            segments.append(entry)
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
        """Translate using Gemini LLM for natural, fluent output."""
        from google import genai

        client = genai.Client(api_key=api_key)

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the source language"

        # Calculate word count guidance for duration matching
        word_count = len(full_text.split())
        duration_hint = ""
        if speech_duration > 0:
            # TTS speaks ~130-150 words/min for most languages
            target_words = int(speech_duration / 60 * 135)
            duration_hint = (
                f"IMPORTANT TIMING CONSTRAINT: The original narration is {int(speech_duration)} seconds long "
                f"({word_count} words). Your {target_name} translation will be spoken by TTS and "
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
                f"Translate the following narration from {source_name} into natural, fluent {target_name}. "
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

        src = self.cfg.source_language if self.cfg.source_language != "auto" else "auto"
        translator = GoogleTranslator(source=src, target=self.cfg.target_language)
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
        """Translate each segment individually, preserving timestamps for sync.

        Priority: OpenAI GPT-4o > Groq Llama 3.3 > Gemini > Google Translate fallback
        """
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

        if openai_key:
            self._report("translate", 0.05, "Using GPT-4o for premium translation...")
            self._translate_segments_openai(segments, openai_key)
        elif groq_key:
            self._report("translate", 0.05, "Using Groq (Llama 3.3 70B) for fast translation...")
            self._translate_segments_groq(segments, groq_key)
        elif gemini_key:
            self._report("translate", 0.05, "Using Gemini for translation...")
            self._translate_segments_gemini(segments, gemini_key)
        else:
            self._report("translate", 0.1, "No API keys found, using Google Translate...")
            self._translate_segments_google(segments)

    def _translate_segments_gemini(self, segments, api_key):
        """Translate segments in numbered batches using Gemini for context-aware output."""
        from google import genai
        client = genai.Client(api_key=api_key)

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the detected language"

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
                f"You are a street-smart dubbing translator who speaks {target_name} like a real person, "
                f"not a textbook. Translate each numbered line from {source_name} to {target_name}.\n\n"
                f"STYLE GUIDE:\n"
                f"- Talk like a REAL {target_name} speaker in everyday life — casual, relatable, human.\n"
                f"- Use COLLOQUIAL / spoken {target_name} with natural slang and filler words people actually use.\n"
                f"- Mix in commonly used English words where native speakers naturally would "
                f"(e.g., 'actually', 'basically', 'so', 'right' stay in English if that's how people talk).\n"
                f"- NEVER sound like a news anchor, textbook, or formal document.\n"
                f"- Match the VIBE — if the original is excited, be excited. If casual, be casual. If serious, be serious.\n"
                f"- The [Xs] shows speaking time — keep translation speakable in roughly that duration.\n"
                f"- Keep proper nouns, brands, and technical terms as-is.\n"
                f"- Short lines stay short. Don't over-explain.\n"
                f"- Output ONLY the numbered {target_name} translations, one per line, matching input numbering.\n\n"
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

    def _translate_segments_groq(self, segments, api_key):
        """Translate segments using Groq (Llama 3.3 70B) — fast, free, context-aware."""
        from groq import Groq
        client = Groq(api_key=api_key)

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the detected language"

        batch_size = 30
        total_batches = (len(segments) + batch_size - 1) // batch_size

        system_msg = (
            f"You are a street-smart dubbing translator who speaks {target_name} like a real person. "
            f"You translate video scripts into colloquial, everyday spoken {target_name} with natural slang — "
            f"the way real people actually talk, NOT textbook language. "
            f"Mix in commonly used English words where native speakers naturally would. "
            f"Keep proper nouns, brands, and technical terms as-is. "
            f"The [Xs] shows speaking time — keep translation speakable in roughly that duration. "
            f"Output ONLY numbered translations, one per line."
        )

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            lines = []
            for i, seg in enumerate(batch):
                duration = seg["end"] - seg["start"]
                lines.append(f"{i+1}. [{duration:.1f}s] {seg['text']}")

            user_msg = (
                f"Translate each line from {source_name} to {target_name}. "
                f"Casual, spoken style. Output ONLY numbered {target_name} translations:\n\n"
                + "\n".join(lines)
            )

            retries = 3
            success = False
            for attempt in range(retries):
                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.7,
                        max_tokens=4096,
                    )
                    result_text = response.choices[0].message.content
                    translations = self._parse_numbered_translations(result_text, len(batch))
                    for i, seg in enumerate(batch):
                        seg["text_translated"] = translations[i] if translations[i] else seg["text"]
                    success = True
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.1 + 0.8 * (batch_idx / total_batches),
                                     f"Groq rate limited, retrying in {wait}s...")
                        time.sleep(wait)

            if not success:
                self._report("translate", 0.1, "Groq failed, using Google Translate for batch...")
                for seg in batch:
                    seg["text_translated"] = self._translate_single_google(seg["text"])

            self._report("translate", 0.1 + 0.9 * ((batch_idx + 1) / total_batches),
                         f"Translated batch {batch_idx + 1}/{total_batches} (Groq)")

    def _translate_segments_openai(self, segments, api_key):
        """Translate segments using OpenAI GPT-4o for highest quality."""
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the detected language"

        batch_size = 30
        total_batches = (len(segments) + batch_size - 1) // batch_size

        system_msg = (
            f"You are a street-smart dubbing translator who speaks {target_name} like a real person. "
            f"You translate video scripts into colloquial, everyday spoken {target_name} with natural slang — "
            f"the way real people actually talk, NOT textbook language. "
            f"Mix in common English words where native {target_name} speakers naturally would. "
            f"Match the vibe/energy of the original. Never sound formal or robotic. "
            f"Keep proper nouns and tech terms as-is. Output ONLY numbered translations matching input numbering."
        )

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            lines = []
            for i, seg in enumerate(batch):
                duration = seg["end"] - seg["start"]
                lines.append(f"{i+1}. [{duration:.1f}s] {seg['text']}")

            user_msg = (
                f"Translate from {source_name} to {target_name}. "
                f"[Xs] = how long the original takes to speak — keep translation speakable in roughly that time.\n\n"
                + "\n".join(lines)
            )

            retries = 3
            success = False
            for attempt in range(retries):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.3,
                    )
                    text = response.choices[0].message.content or ""
                    translations = self._parse_numbered_translations(text, len(batch))
                    for i, seg in enumerate(batch):
                        seg["text_translated"] = translations[i] if translations[i] else seg["text"]
                    success = True
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.1 + 0.8 * (batch_idx / total_batches),
                                     f"GPT-4o rate limited, retrying in {wait}s...")
                        time.sleep(wait)

            if not success:
                # Fall back to Groq > Gemini > Google Translate
                groq_key = os.environ.get("GROQ_API_KEY", "").strip()
                gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
                if groq_key:
                    self._report("translate", 0.1, "GPT-4o failed, falling back to Groq...")
                    for seg in batch:
                        seg["text_translated"] = seg.get("text", "")
                    self._translate_segments_groq(batch, groq_key)
                elif gemini_key:
                    self._report("translate", 0.1, "GPT-4o failed, falling back to Gemini...")
                    for seg in batch:
                        seg["text_translated"] = seg.get("text", "")
                    self._translate_segments_gemini(batch, gemini_key)
                else:
                    for seg in batch:
                        seg["text_translated"] = self._translate_single_google(seg["text"])

            self._report("translate", 0.1 + 0.9 * ((batch_idx + 1) / total_batches),
                         f"Translated batch {batch_idx + 1}/{total_batches} (GPT-4o)")

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
            src = self.cfg.source_language if self.cfg.source_language != "auto" else "auto"
            translator = GoogleTranslator(source=src, target=self.cfg.target_language)
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
        """Natural-flow TTS: generate at natural speed, place at original timestamps.

        No speed manipulation — speech sounds completely natural.
        If a segment runs longer than its slot, it simply overlaps into the next gap.
        """
        import edge_tts
        voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        # Generate all TTS at natural rate
        async def tts_generate():
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
                    pass

        asyncio.run(tts_generate())

        # Convert to WAV (no speed changes)
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

    # ── Natural TTS + Video sync ────────────────────────────────────────
    # Languages that Chatterbox TTS can pronounce well (English only for now)
    CHATTERBOX_SUPPORTED_LANGS = {"en"}

    def _generate_tts_natural(self, segments):
        """Generate TTS at natural speed. Uses first enabled engine in priority order.

        Chatterbox is English-only — for other languages it auto-falls back to
        ElevenLabs (multilingual) or Edge-TTS (which has native voices for 70+ languages).
        """
        target = self.cfg.target_language

        if self.cfg.use_chatterbox:
            if target in self.CHATTERBOX_SUPPORTED_LANGS:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError("No CUDA GPU available")
                    self._report("synthesize", 0.05, "Using Chatterbox TTS (GPU, human-like voice)...")
                    return self._tts_chatterbox(segments)
                except Exception as e:
                    self._report("synthesize", 0.05,
                                 f"Chatterbox failed ({e}) — falling back to Edge-TTS...")
            else:
                target_name = LANGUAGE_NAMES.get(target, target)
                self._report("synthesize", 0.05,
                             f"Chatterbox is English-only — switching to Edge-TTS for {target_name}...")

        if self.cfg.use_elevenlabs:
            elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
            if elevenlabs_key:
                self._report("synthesize", 0.05, "Using ElevenLabs for human-like voice...")
                return self._tts_elevenlabs(segments, elevenlabs_key)
            if not self.cfg.use_chatterbox:
                raise RuntimeError("ElevenLabs enabled but ELEVENLABS_API_KEY not set in .env")

        # Fall through to Edge-TTS (supports 70+ languages with native voices)
        voice = self.cfg.tts_voice
        target_name = LANGUAGE_NAMES.get(target, target)
        self._report("synthesize", 0.05, f"Using Edge-TTS ({voice}) for {target_name}...")
        return self._tts_edge(segments)

    def _tts_chatterbox(self, segments):
        """Generate TTS using Chatterbox — free, local, human-like AI voice."""
        import torch
        import torchaudio
        from chatterbox.tts import ChatterboxTTS

        self._report("synthesize", 0.05, "Loading Chatterbox model on GPU...")
        model = ChatterboxTTS.from_pretrained(device="cuda")

        tts_data = []
        for i, seg in enumerate(segments):
            text = seg.get("text_translated", seg["text"]).strip()
            if not text:
                continue

            wav_path = self.cfg.work_dir / f"tts_{i:04d}.wav"

            try:
                wav_tensor = model.generate(text)
                # Save as WAV — Chatterbox outputs at 24kHz
                torchaudio.save(str(wav_path), wav_tensor.cpu(), model.sr)

                # Resample to our pipeline's sample rate
                if model.sr != self.SAMPLE_RATE:
                    resampled = self.cfg.work_dir / f"tts_{i:04d}_rs.wav"
                    subprocess.run(
                        [self._ffmpeg, "-y", "-i", str(wav_path),
                         "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                         str(resampled)],
                        check=True, capture_output=True,
                    )
                    wav_path.unlink(missing_ok=True)
                    resampled.rename(wav_path)

            except Exception as e:
                self._report("synthesize", 0.1 + 0.8 * ((i + 1) / len(segments)),
                             f"Chatterbox error on seg {i+1}: {e}, skipping...")
                continue

            if not wav_path.exists() or wav_path.stat().st_size == 0:
                continue

            tts_dur = self._get_duration(wav_path)
            tts_data.append({
                "start": seg["start"],
                "end": seg["end"],
                "wav": wav_path,
                "duration": tts_dur,
            })
            self._report(
                "synthesize",
                0.1 + 0.8 * ((i + 1) / len(segments)),
                f"Synthesized {i + 1}/{len(segments)} segments (Chatterbox)...",
            )

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        return tts_data

    def _tts_elevenlabs(self, segments, api_key):
        """Generate TTS using ElevenLabs — paid API, most human-like."""
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=api_key)

        voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
        if not voice_id:
            voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel — clear, natural

        model_id = "eleven_multilingual_v2"

        tts_data = []
        for i, seg in enumerate(segments):
            text = seg.get("text_translated", seg["text"]).strip()
            if not text:
                continue

            mp3 = self.cfg.work_dir / f"tts_{i:04d}.mp3"

            try:
                audio_gen = client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id=model_id,
                    output_format="mp3_44100_128",
                )
                with open(mp3, "wb") as f:
                    for chunk in audio_gen:
                        f.write(chunk)
            except Exception:
                try:
                    asyncio.run(self._edge_tts_single(text, mp3))
                except Exception:
                    continue

            if not mp3.exists() or mp3.stat().st_size == 0:
                continue

            wav = self.cfg.work_dir / f"tts_{i:04d}.wav"
            subprocess.run(
                [self._ffmpeg, "-y", "-i", str(mp3),
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 str(wav)],
                check=True, capture_output=True,
            )
            mp3.unlink(missing_ok=True)
            tts_dur = self._get_duration(wav)
            tts_data.append({
                "start": seg["start"],
                "end": seg["end"],
                "wav": wav,
                "duration": tts_dur,
            })
            self._report(
                "synthesize",
                0.1 + 0.8 * ((i + 1) / len(segments)),
                f"Synthesized {i + 1}/{len(segments)} segments (ElevenLabs)...",
            )

        return tts_data

    async def _edge_tts_single(self, text, mp3_path):
        """Generate a single segment with edge-tts."""
        import edge_tts
        comm = edge_tts.Communicate(text, self.cfg.tts_voice, rate=self.cfg.tts_rate)
        await comm.save(str(mp3_path))

    def _tts_edge(self, segments):
        """Generate TTS using edge-tts (free Microsoft voices)."""
        import edge_tts
        voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        async def generate():
            for i, seg in enumerate(segments):
                text = seg.get("text_translated", seg["text"]).strip()
                if not text:
                    continue
                mp3 = self.cfg.work_dir / f"tts_{i:04d}.mp3"
                try:
                    comm = edge_tts.Communicate(text, voice, rate=rate)
                    await comm.save(str(mp3))
                    seg["_tts_mp3"] = mp3
                except Exception:
                    pass
                self._report(
                    "synthesize",
                    0.1 + 0.8 * ((i + 1) / len(segments)),
                    f"Synthesized {i + 1}/{len(segments)} segments...",
                )

        asyncio.run(generate())

        tts_data = []
        for i, seg in enumerate(segments):
            mp3 = seg.pop("_tts_mp3", None)
            if not mp3 or not mp3.exists():
                continue
            wav = self.cfg.work_dir / f"tts_{i:04d}.wav"
            subprocess.run(
                [self._ffmpeg, "-y", "-i", str(mp3),
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 str(wav)],
                check=True, capture_output=True,
            )
            mp3.unlink(missing_ok=True)
            tts_dur = self._get_duration(wav)
            tts_data.append({
                "start": seg["start"],
                "end": seg["end"],
                "wav": wav,
                "duration": tts_dur,
            })

        return tts_data

    def _build_video_synced(self, video_path, audio_raw, tts_data, total_video_duration):
        """Adjust video speed per-segment to match natural TTS duration.

        Instead of changing audio speed, adjust video speed:
        - TTS longer than original → slow video (scene lingers)
        - TTS shorter than original → speed video up slightly
        - Gaps between speech play at normal speed
        """
        # Build sections: alternating gaps and speech segments
        sections = []
        current_pos = 0.0

        for tts in tts_data:
            seg_start = tts["start"]
            seg_end = tts["end"]
            tts_dur = tts["duration"]
            original_dur = seg_end - seg_start

            # Gap before this segment
            if seg_start > current_pos + 0.05:
                sections.append({
                    "type": "gap",
                    "video_start": current_pos,
                    "video_end": seg_start,
                })

            # Speech segment — compute video speed factor
            speed_factor = (tts_dur / original_dur) if original_dur > 0.1 else 1.0
            # Clamp to avoid extreme distortion
            speed_factor = max(0.5, min(speed_factor, 2.5))

            sections.append({
                "type": "speech",
                "video_start": seg_start,
                "video_end": seg_end,
                "speed_factor": speed_factor,
                "tts_wav": tts["wav"],
                "tts_dur": tts_dur,
            })

            current_pos = seg_end

        # Trailing gap after last speech
        if current_pos < total_video_duration - 0.05:
            sections.append({
                "type": "gap",
                "video_start": current_pos,
                "video_end": total_video_duration,
            })

        # Create video clips for each section
        num_sections = len(sections)
        clip_paths = []

        for idx, sec in enumerate(sections):
            self._report("assemble",
                         0.1 + 0.6 * (idx / max(num_sections, 1)),
                         f"Syncing section {idx + 1}/{num_sections}...")

            clip = self.cfg.work_dir / f"vsync_{idx:04d}.mp4"
            vs = sec["video_start"]
            ve = sec["video_end"]
            dur = ve - vs

            if dur < 0.05:
                continue

            factor = sec.get("speed_factor", 1.0)

            # Always re-encode for consistent concat (mixed copy+encode breaks concat demuxer)
            # factor > 1 → slow motion, factor < 1 → fast forward, ~1 → normal
            subprocess.run(
                [self._ffmpeg, "-y",
                 "-ss", f"{vs:.3f}", "-i", str(video_path),
                 "-t", f"{dur:.3f}",
                 "-filter:v", f"setpts={factor:.6f}*PTS",
                 "-an",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 str(clip)],
                check=True, capture_output=True,
            )

            clip_paths.append(clip)

        if not clip_paths:
            raise RuntimeError("No video sections produced")

        # Concatenate all video clips
        self._report("assemble", 0.75, "Joining video sections...")
        synced_video = self.cfg.work_dir / "video_synced.mp4"
        if len(clip_paths) == 1:
            shutil.copy2(clip_paths[0], synced_video)
        else:
            self._concatenate_videos(clip_paths, synced_video)

        # Build TTS audio timeline matching the adjusted video timing
        new_pos = 0.0
        audio_segments = []
        for sec in sections:
            dur = sec["video_end"] - sec["video_start"]
            if dur < 0.05:
                continue
            if sec["type"] == "speech":
                audio_segments.append({
                    "start": new_pos,
                    "wav": sec["tts_wav"],
                    "duration": sec["tts_dur"],
                })
                new_pos += sec["tts_dur"]
            else:
                new_pos += dur  # gaps keep original duration

        self._report("assemble", 0.85, "Building audio timeline...")
        synced_audio = self._build_timeline(audio_segments, new_pos, prefix="synced_")

        # Mix original audio at low volume if requested
        if self.cfg.mix_original:
            synced_audio = self._mix_audio(audio_raw, synced_audio, self.cfg.original_volume)

        # Mux final video + audio
        self._report("assemble", 0.90, "Muxing final video...")
        self._mux_replace_audio(synced_video, synced_audio, self.cfg.output_path)

    @staticmethod
    def _parse_tts_rate(rate_str: str) -> int:
        """Parse edge-tts rate string like '-5%' or '+20%' to integer."""
        return int(rate_str.replace("%", "").replace("+", ""))

    def _time_stretch(self, wav_path: Path, ratio: float, output_path: Path) -> Path:
        """Time-stretch audio (ratio > 1 = speed up). Tries rubberband then atempo."""
        # Try ffmpeg rubberband filter first (better pitch preservation)
        try:
            subprocess.run(
                [self._ffmpeg, "-y", "-i", str(wav_path),
                 "-filter:a", f"rubberband=tempo={ratio:.4f}",
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 str(output_path)],
                check=True, capture_output=True,
            )
            return output_path
        except subprocess.CalledProcessError:
            pass  # rubberband not available, fall back to atempo

        # Fallback: atempo filter (chain for ratios > 2.0)
        tempo = ratio
        filters = []
        while tempo > 2.0:
            filters.append("atempo=2.0")
            tempo /= 2.0
        filters.append(f"atempo={tempo:.4f}")
        subprocess.run(
            [self._ffmpeg, "-y", "-i", str(wav_path),
             "-filter:a", ",".join(filters),
             "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
             str(output_path)],
            check=True, capture_output=True,
        )
        return output_path

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
