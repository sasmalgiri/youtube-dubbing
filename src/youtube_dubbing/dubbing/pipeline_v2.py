"""
Enhanced YouTube Dubbing Pipeline v2
=====================================
Features:
- Progress bars for all steps
- Time-aligned TTS with silence padding
- Mix original audio option
- Better error handling
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Iterable

import asyncio
import re
import shutil
import subprocess
import struct
import wave
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

from youtube_dubbing.subtitles.srt import write_srt


@dataclass
class PipelineConfig:
    source: str
    work_dir: Path
    output_path: Path

    # Unused in English-only MVP
    target_language: Optional[str] = None
    voice_dir: Optional[Path] = None

    asr_model: str = "small"
    transcribe_only: bool = False

    # TTS settings
    tts_voice: str = "en-US-AriaNeural"
    tts_rate: str = "+0%"

    # Enhanced options
    mix_original: bool = False       # Blend original audio with TTS
    original_volume: float = 0.15    # Volume of original audio when mixing (0.0-1.0)
    time_aligned: bool = True        # Align TTS to original timestamps


class Pipeline:
    """Enhanced dubbing pipeline with progress tracking and time-aligned TTS."""

    SAMPLE_RATE = 48000
    N_CHANNELS = 2

    def __init__(self, cfg: PipelineConfig, console: Optional[Console] = None):
        self.cfg = cfg
        self.console = console or Console()
        self.input_dir = Path("input")
        self.output_dir = Path("output")
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def run(self):
        """Execute the full dubbing pipeline."""
        self.console.print(Panel.fit(
            "[bold cyan]YouTube Dubbing Pipeline v2[/bold cyan]\n"
            "[dim]Enhanced with progress tracking & time-aligned TTS[/dim]",
            box=box.ROUNDED
        ))

        try:
            self._ensure_ffmpeg()

            # Step 1: Ingest source
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                progress.add_task("Ingesting source...", total=None)
                video_path = self._ingest_source(self.cfg.source)
            self.console.print(f"[green]✓[/green] Source: [cyan]{video_path.name}[/cyan]")

            # Step 2: Extract audio
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                progress.add_task("Extracting audio...", total=None)
                audio_raw = self._extract_audio(video_path)
            self.console.print(f"[green]✓[/green] Audio extracted")

            # Step 3: Transcribe
            self.console.print(f"[blue]Transcribing with faster-whisper ({self.cfg.asr_model})...[/blue]")
            segments = self._transcribe_with_progress(audio_raw, model_size=self.cfg.asr_model)
            self.console.print(f"[green]✓[/green] Transcribed {len(segments)} segments")

            # Write SRT
            srt_en = self.cfg.work_dir / "transcript_en.srt"
            write_srt(segments, srt_en)
            self.console.print(f"[green]✓[/green] SRT saved: [cyan]{srt_en}[/cyan]")

            if self.cfg.transcribe_only:
                self.console.print(Panel("[green]Transcription complete![/green]", box=box.ROUNDED))
                return

            # Check for speech
            text_segments = [s for s in segments if s.get("text", "").strip()]
            if not text_segments:
                self.console.print("[yellow]⚠ No speech detected in video. Cannot generate TTS.[/yellow]")
                return

            # Step 4: TTS Synthesis
            self.console.print(f"[blue]Synthesizing TTS (voice: {self.cfg.tts_voice})...[/blue]")
            if self.cfg.time_aligned:
                tts_wav = self._tts_time_aligned(segments)
            else:
                tts_wav = self._tts_concatenated(segments)

            if not tts_wav.exists():
                self.console.print("[red]✗ TTS synthesis failed[/red]")
                return
            self.console.print(f"[green]✓[/green] TTS audio generated")

            # Step 5: Mix or replace audio
            final_audio = tts_wav
            if self.cfg.mix_original:
                self.console.print(f"[blue]Mixing original audio ({self.cfg.original_volume:.0%} volume)...[/blue]")
                final_audio = self._mix_audio(audio_raw, tts_wav, self.cfg.original_volume)
                self.console.print(f"[green]✓[/green] Audio mixed")

            # Step 6: Mux into video
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                progress.add_task("Muxing audio into video...", total=None)
                self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
                self._mux_replace_audio(video_path, final_audio, self.cfg.output_path)

            self.console.print(Panel(
                f"[green bold]✓ Done![/green bold]\n"
                f"Output: [cyan]{self.cfg.output_path}[/cyan]",
                box=box.ROUNDED
            ))

        except FileNotFoundError as e:
            self.console.print(f"[red]✗ File not found:[/red] {e}")
            raise
        except RuntimeError as e:
            self.console.print(f"[red]✗ Error:[/red] {e}")
            raise
        except Exception as e:
            self.console.print(f"[red]✗ Unexpected error:[/red] {e}")
            raise

    def _ensure_ffmpeg(self):
        """Ensure ffmpeg is available, auto-adding WinGet path on Windows."""
        if shutil.which("ffmpeg") is None and sys.platform == "win32":
            localappdata = os.environ.get("LOCALAPPDATA", "")
            winget_ffmpeg = Path(localappdata) / "Microsoft" / "WinGet" / "Packages"
            if winget_ffmpeg.exists():
                for pkg in winget_ffmpeg.iterdir():
                    if pkg.name.startswith("Gyan.FFmpeg"):
                        for sub in pkg.iterdir():
                            if sub.is_dir():
                                bin_path = sub / "bin"
                                if bin_path.exists() and (bin_path / "ffmpeg.exe").exists():
                                    os.environ["PATH"] = str(bin_path) + os.pathsep + os.environ.get("PATH", "")
                                    self.console.print(f"[dim]Added FFmpeg: {bin_path}[/dim]")
                                    break

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "FFmpeg not found!\n"
                "  Install: winget install Gyan.FFmpeg\n"
                "  Or download: https://ffmpeg.org/download.html"
            )

    def _ingest_source(self, src: str) -> Path:
        """Download URL or locate local file."""
        if re.match(r"^https?://", src):
            out_tpl = str(self.cfg.work_dir / "source.%(ext)s")
            try:
                subprocess.run(
                    ["yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4", "-o", out_tpl, src],
                    check=True, capture_output=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"yt-dlp failed: {e.stderr.decode() if e.stderr else 'Unknown error'}\n"
                    "Tips:\n"
                    "  - Ensure you own the content\n"
                    "  - Try: yt-dlp --cookies-from-browser chrome <url>"
                ) from e
            mp4 = list(self.cfg.work_dir.glob("source.mp4"))
            return mp4[0] if mp4 else list(self.cfg.work_dir.glob("source.*"))[0]

        p = Path(src)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.is_dir():
            p = self._pick_video_in_dir(p)
        if not p or not p.exists():
            raise FileNotFoundError(f"Source not found: {src}")
        return p

    def _pick_video_in_dir(self, d: Path) -> Optional[Path]:
        """Find first video file in directory."""
        for ext in (".mp4", ".mkv", ".mov", ".webm", ".m4v"):
            found = sorted(d.glob(f"*{ext}"))
            if found:
                return found[0]
        return None

    def _extract_audio(self, video_path: Path) -> Path:
        """Extract audio to WAV."""
        wav = self.cfg.work_dir / "audio_raw.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", str(self.N_CHANNELS), "-ar", str(self.SAMPLE_RATE),
            "-acodec", "pcm_s16le", str(wav)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return wav

    def _transcribe_with_progress(self, wav_path: Path, model_size: str = "small") -> List[Dict]:
        """Transcribe with progress indication."""
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        seg_iter, info = model.transcribe(str(wav_path), vad_filter=True)

        segments: List[Dict] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            # We don't know total segments upfront, so use indeterminate progress
            task = progress.add_task("Processing segments...", total=None)
            for seg in seg_iter:
                segments.append({
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text.strip(),
                })
                progress.update(task, description=f"Transcribed {len(segments)} segments...")

        return segments

    def _tts_time_aligned(self, segments: List[Dict]) -> Path:
        """
        Synthesize TTS with time alignment - each segment starts at its original timestamp.
        Gaps are filled with silence for proper lip-sync.
        """
        import edge_tts

        out_wav = self.cfg.work_dir / "tts_aligned.wav"
        voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        async def synthesize_segment(text: str, seg_out: Path):
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(str(seg_out))

        async def run_all():
            seg_data: List[tuple] = []  # (start_time, wav_path)

            text_segments = [s for s in segments if s.get("text", "").strip()]

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("Synthesizing...", total=len(text_segments))

                for i, seg in enumerate(text_segments):
                    text = seg.get("text", "").strip()
                    start_time = seg.get("start", 0.0)

                    seg_mp3 = self.cfg.work_dir / f"tts_seg_{i:04d}.mp3"
                    seg_wav = self.cfg.work_dir / f"tts_seg_{i:04d}.wav"

                    await synthesize_segment(text, seg_mp3)

                    # Convert to WAV
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(seg_mp3),
                         "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                         str(seg_wav)],
                        check=True, capture_output=True
                    )
                    seg_mp3.unlink(missing_ok=True)
                    seg_data.append((start_time, seg_wav))
                    progress.update(task, advance=1)

            # Build time-aligned audio
            self._build_aligned_wav(seg_data, out_wav)

            # Cleanup
            for _, wav_path in seg_data:
                wav_path.unlink(missing_ok=True)

        asyncio.run(run_all())
        return out_wav

    def _build_aligned_wav(self, seg_data: List[tuple], out_wav: Path):
        """Build a WAV file with segments placed at their timestamps."""
        if not seg_data:
            return

        # Find total duration needed
        max_end = 0.0
        for start_time, wav_path in seg_data:
            with wave.open(str(wav_path), "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
                max_end = max(max_end, start_time + duration)

        total_samples = int(max_end * self.SAMPLE_RATE) + self.SAMPLE_RATE  # +1 sec buffer
        audio_buffer = [0] * (total_samples * self.N_CHANNELS)

        for start_time, wav_path in seg_data:
            start_sample = int(start_time * self.SAMPLE_RATE)
            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                samples = struct.unpack(f"<{len(frames)//2}h", frames)

                for j, sample in enumerate(samples):
                    idx = start_sample * self.N_CHANNELS + j
                    if idx < len(audio_buffer):
                        # Mix (add with clipping)
                        audio_buffer[idx] = max(-32768, min(32767, audio_buffer[idx] + sample))

        # Write output
        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(self.N_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(struct.pack(f"<{len(audio_buffer)}h", *audio_buffer))

    def _tts_concatenated(self, segments: List[Dict]) -> Path:
        """Simple concatenation of TTS segments (original behavior)."""
        import edge_tts

        out_wav = self.cfg.work_dir / "tts_en.wav"
        voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        async def synthesize_segment(text: str, seg_out: Path):
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(str(seg_out))

        async def run_all():
            seg_wavs: List[Path] = []
            text_segments = [s for s in segments if s.get("text", "").strip()]

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("Synthesizing...", total=len(text_segments))

                for i, seg in enumerate(text_segments):
                    text = seg.get("text", "").strip()
                    seg_mp3 = self.cfg.work_dir / f"tts_seg_{i:04d}.mp3"
                    seg_wav = self.cfg.work_dir / f"tts_seg_{i:04d}.wav"

                    await synthesize_segment(text, seg_mp3)
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(seg_mp3),
                         "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                         str(seg_wav)],
                        check=True, capture_output=True
                    )
                    seg_wavs.append(seg_wav)
                    seg_mp3.unlink(missing_ok=True)
                    progress.update(task, advance=1)

            # Concatenate
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

    def _mix_audio(self, original: Path, tts: Path, original_vol: float) -> Path:
        """Mix original audio (lowered) with TTS audio."""
        mixed = self.cfg.work_dir / "audio_mixed.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(tts),
            "-i", str(original),
            "-filter_complex",
            f"[1:a]volume={original_vol}[orig];[0:a][orig]amix=inputs=2:duration=first:dropout_transition=2[out]",
            "-map", "[out]",
            "-ar", str(self.SAMPLE_RATE),
            "-ac", str(self.N_CHANNELS),
            str(mixed)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return mixed

    def _mux_replace_audio(self, video_path: Path, audio_path: Path, output_path: Path):
        """Replace video's audio track."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)


async def list_voices(language_filter: str = "en"):
    """List available edge-tts voices."""
    import edge_tts
    voices = await edge_tts.list_voices()
    if language_filter:
        voices = [v for v in voices if v.get("Locale", "").startswith(language_filter)]
    return voices


def print_voices(console: Console, language_filter: str = "en"):
    """Print available voices in a nice table."""
    from rich.table import Table

    voices = asyncio.run(list_voices(language_filter))

    table = Table(title=f"Available Edge-TTS Voices ({language_filter}*)")
    table.add_column("Voice Name", style="cyan")
    table.add_column("Gender", style="magenta")
    table.add_column("Locale", style="green")

    for v in sorted(voices, key=lambda x: x.get("ShortName", "")):
        table.add_row(
            v.get("ShortName", ""),
            v.get("Gender", ""),
            v.get("Locale", "")
        )

    console.print(table)
    console.print(f"\n[dim]Use --tts-voice <name> to select a voice[/dim]")


__all__ = ["Pipeline", "PipelineConfig", "print_voices", "list_voices"]
