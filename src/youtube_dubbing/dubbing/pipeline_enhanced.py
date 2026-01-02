from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Iterable

import asyncio
import re
import shutil
import subprocess
import struct
import wave

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

from youtube_dubbing.subtitles.srt import write_srt


@dataclass
class PipelineConfig:
    source: str
    work_dir: Path
    output_path: Path

    # Keep but unused in English-only MVP
    target_language: Optional[str] = None
    voice_dir: Optional[Path] = None

    asr_model: str = "small"  # default to small for CPU
    transcribe_only: bool = False

    # English-only AI voice (Microsoft Edge Neural)
    tts_voice: str = "en-US-AriaNeural"
    tts_rate: str = "+0%"
    
    # Enhanced options
    keep_original_audio: bool = False  # Mix original audio (lowered) with TTS
    original_audio_volume: float = 0.15  # Volume of original audio when mixing (0.0-1.0)
    cleanup_temp_files: bool = True  # Remove intermediate files after completion


class Pipeline:
    def __init__(self, cfg: PipelineConfig, console: Optional[Console] = None):
        self.cfg = cfg
        self.console = console or Console()
        self.input_dir = Path("input")
        self.output_dir = Path("output")
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def run(self):
        self._print_header()
        self._ensure_ffmpeg()

        video_path = self._ingest_source(self.cfg.source)
        video_duration = self._get_video_duration(video_path)
        audio_raw = self._extract_audio(video_path)

        segments = self._transcribe_with_progress(audio_raw, model_size=self.cfg.asr_model)

        srt_en = self.cfg.work_dir / "transcript_en.srt"
        write_srt(segments, srt_en)
        self.console.print(f"[green]✓ Wrote English SRT:[/green] {srt_en}")

        if self.cfg.transcribe_only:
            self._print_summary(video_path, srt_en, segments)
            return

        # Check if we have any text to synthesize
        text_segments = [s for s in segments if s.get("text", "").strip()]
        if not text_segments:
            self.console.print("[yellow]⚠ No speech detected in video. Cannot generate TTS.[/yellow]")
            self._print_summary(video_path, srt_en, segments)
            return

        # === AI TTS (English) with time alignment ===
        tts_wav = self._tts_edge_aligned(segments, video_duration, voice=self.cfg.tts_voice, rate=self.cfg.tts_rate)

        if not tts_wav.exists():
            self.console.print("[red]✗ TTS synthesis failed to produce output.[/red]")
            return

        # === Mux TTS audio into video ===
        self._mux_with_progress(video_path, audio_raw, tts_wav, self.cfg.output_path)

        # Cleanup temp files if requested
        if self.cfg.cleanup_temp_files:
            self._cleanup_temp_files()

        self._print_completion(self.cfg.output_path, segments)

    def _print_header(self):
        """Print a styled header."""
        panel = Panel.fit(
            "[bold cyan]YouTube Dubbing Pipeline[/bold cyan]\n"
            "[dim]English AI Voice Dubbing • CPU Mode[/dim]",
            box=box.DOUBLE_EDGE,
            border_style="cyan"
        )
        self.console.print(panel)
        self.console.print()

    def _print_summary(self, video_path: Path, srt_path: Path, segments: List[Dict]):
        """Print a summary table."""
        table = Table(title="Summary", box=box.ROUNDED)
        table.add_column("Item", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Source Video", str(video_path.name))
        table.add_row("Transcript", str(srt_path))
        table.add_row("Segments Found", str(len(segments)))
        
        if segments:
            total_words = sum(len(s.get("text", "").split()) for s in segments)
            table.add_row("Total Words", str(total_words))
        
        self.console.print(table)

    def _print_completion(self, output_path: Path, segments: List[Dict]):
        """Print completion message with statistics."""
        self.console.print()
        panel = Panel(
            f"[bold green]✓ Dubbing Complete![/bold green]\n\n"
            f"[white]Output:[/white] [cyan]{output_path}[/cyan]\n"
            f"[white]Segments dubbed:[/white] [cyan]{len(segments)}[/cyan]\n"
            f"[white]Voice:[/white] [cyan]{self.cfg.tts_voice}[/cyan]",
            box=box.DOUBLE_EDGE,
            border_style="green",
            title="Success",
            title_align="left"
        )
        self.console.print(panel)

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True, check=True
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _ensure_ffmpeg(self):
        """Ensure ffmpeg is available, auto-adding WinGet path on Windows if needed."""
        import os
        import sys
        
        # Try to find ffmpeg in common Windows locations if not on PATH
        if shutil.which("ffmpeg") is None and sys.platform == "win32":
            localappdata = os.environ.get("LOCALAPPDATA", "")
            winget_ffmpeg = Path(localappdata) / "Microsoft" / "WinGet" / "Packages"
            if winget_ffmpeg.exists():
                for pkg in winget_ffmpeg.iterdir():
                    if pkg.name.startswith("Gyan.FFmpeg"):
                        bin_path = pkg / "ffmpeg-8.0-full_build" / "bin"
                        if not bin_path.exists():
                            for sub in pkg.iterdir():
                                if sub.is_dir() and "ffmpeg" in sub.name.lower():
                                    bin_path = sub / "bin"
                                    break
                        if bin_path.exists() and (bin_path / "ffmpeg.exe").exists():
                            os.environ["PATH"] = str(bin_path) + os.pathsep + os.environ.get("PATH", "")
                            self.console.print(f"[dim]Added FFmpeg to PATH: {bin_path}[/dim]")
                            break
        
        try:
            if shutil.which("ffmpeg") is None:
                raise RuntimeError("ffmpeg not on PATH")
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as e:
            raise RuntimeError(
                "FFmpeg is required but not available on PATH.\n"
                "Install via: winget install Gyan.FFmpeg\n"
                "Or download from: https://ffmpeg.org/download.html"
            ) from e

    def _ingest_source(self, src: str) -> Path:
        if re.match(r"^https?://", src):
            out_tpl = str(self.cfg.work_dir / "source.%(ext)s")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                progress.add_task("[cyan]Downloading video with yt-dlp...", total=None)
                try:
                    subprocess.run(
                        ["yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4", "-o", out_tpl, src],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(
                        "yt-dlp failed to download the URL.\n"
                        "• The video may be unavailable or require authentication\n"
                        "• Ensure you own the content\n"
                        "• Try: yt-dlp --cookies-from-browser chrome <url>"
                    ) from e
            mp4 = list(self.cfg.work_dir.glob("source.mp4"))
            if mp4:
                self.console.print(f"[green]✓ Downloaded:[/green] {mp4[0].name}")
                return mp4[0]
            files = list(self.cfg.work_dir.glob("source.*"))
            if not files:
                raise FileNotFoundError("yt-dlp did not produce a file.")
            return files[0]
        else:
            p = Path(src)
            if p.is_dir():
                picked = self._pick_video_in_dir(p)
                if picked:
                    return picked
                raise FileNotFoundError(f"No video files found in directory: {p}")
            if not p.is_absolute():
                p = Path.cwd() / p
            if p.exists():
                self.console.print(f"[green]✓ Source:[/green] {p.name}")
                return p
            parent = p.parent if p.parent.exists() else None
            if parent:
                picked = self._pick_video_in_dir(parent)
                if picked:
                    self.console.print(f"[yellow]Source '{src}' not found. Using: {picked}[/yellow]")
                    return picked
            raise FileNotFoundError(f"Source not found: {p}")

    def _pick_video_in_dir(self, d: Path) -> Optional[Path]:
        exts: Iterable[str] = (".mp4", ".mkv", ".mov", ".webm", ".m4v")
        for ext in exts:
            found = sorted(d.glob(f"*{ext}"))
            if found:
                return found[0]
        return None

    def _extract_audio(self, video_path: Path) -> Path:
        wav = self.cfg.work_dir / "audio_raw.wav"
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "2", "-ar", "48000", "-acodec", "pcm_s16le", str(wav)]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            progress.add_task("[cyan]Extracting audio...", total=None)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        self.console.print(f"[green]✓ Audio extracted[/green]")
        return wav

    def _transcribe_with_progress(self, wav_path: Path, model_size: str = "small") -> List[Dict]:
        """Transcribe with a progress indicator."""
        from faster_whisper import WhisperModel
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task(f"[cyan]Loading Whisper model ({model_size})...", total=None)
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            
            progress.update(task, description="[cyan]Transcribing audio (this may take a while)...")
            seg_iter, info = model.transcribe(str(wav_path), vad_filter=True)
            
            segments: List[Dict] = []
            for seg in seg_iter:
                segments.append({
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text.strip(),
                })
        
        self.console.print(f"[green]✓ Transcribed {len(segments)} segments[/green]")
        return segments

    def _tts_edge_aligned(self, segments: List[Dict], video_duration: float, voice: str, rate: str) -> Path:
        """
        Synthesize TTS audio aligned to original segment timestamps.
        Creates a full-length audio file with TTS at correct positions.
        """
        import edge_tts

        out_wav = self.cfg.work_dir / "tts_en.wav"
        sample_rate = 48000
        n_channels = 2
        bytes_per_sample = 2

        # Calculate total samples needed
        total_samples = int(video_duration * sample_rate) if video_duration > 0 else int(segments[-1]["end"] * sample_rate + sample_rate)
        
        # Create silence buffer for the full duration
        full_audio = bytearray(total_samples * n_channels * bytes_per_sample)

        async def synthesize_segment(text: str, seg_mp3: Path):
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(str(seg_mp3))

        async def run_all():
            text_segments = [(i, seg) for i, seg in enumerate(segments) if seg.get("text", "").strip()]
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task(f"[cyan]Synthesizing TTS ({voice})...", total=len(text_segments))
                
                for i, seg in text_segments:
                    text = seg.get("text", "").strip()
                    start_time = seg["start"]
                    end_time = seg["end"]
                    
                    seg_mp3 = self.cfg.work_dir / f"tts_seg_{i:04d}.mp3"
                    seg_wav = self.cfg.work_dir / f"tts_seg_{i:04d}.wav"
                    
                    try:
                        await synthesize_segment(text, seg_mp3)
                        
                        # Convert to WAV
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", str(seg_mp3), "-ar", str(sample_rate), "-ac", str(n_channels), str(seg_wav)],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                        
                        # Read the synthesized segment
                        with wave.open(str(seg_wav), "rb") as wf:
                            seg_frames = wf.readframes(wf.getnframes())
                        
                        # Calculate position in full audio
                        start_sample = int(start_time * sample_rate)
                        start_byte = start_sample * n_channels * bytes_per_sample
                        
                        # Copy segment audio to correct position (overwrite silence)
                        end_byte = min(start_byte + len(seg_frames), len(full_audio))
                        copy_len = end_byte - start_byte
                        if copy_len > 0:
                            full_audio[start_byte:end_byte] = seg_frames[:copy_len]
                        
                        # Cleanup
                        seg_mp3.unlink(missing_ok=True)
                        seg_wav.unlink(missing_ok=True)
                        
                    except Exception as e:
                        self.console.print(f"[yellow]Warning: TTS failed for segment {i}: {e}[/yellow]")
                    
                    progress.update(task, advance=1)

            # Write the full aligned audio
            with wave.open(str(out_wav), "wb") as wf:
                wf.setnchannels(n_channels)
                wf.setsampwidth(bytes_per_sample)
                wf.setframerate(sample_rate)
                wf.writeframes(bytes(full_audio))

        asyncio.run(run_all())
        self.console.print(f"[green]✓ TTS audio generated (time-aligned)[/green]")
        return out_wav

    def _mux_with_progress(self, video_path: Path, original_audio: Path, tts_audio: Path, output_path: Path):
        """Mux TTS audio into video, optionally mixing with original audio."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            progress.add_task("[cyan]Muxing audio into video...", total=None)
            
            if self.cfg.keep_original_audio:
                # Mix original (lowered) + TTS audio
                vol = self.cfg.original_audio_volume
                filter_complex = f"[1:a]volume={vol}[orig];[2:a][orig]amix=inputs=2:duration=longest[aout]"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-i", str(original_audio),
                    "-i", str(tts_audio),
                    "-filter_complex", filter_complex,
                    "-map", "0:v:0",
                    "-map", "[aout]",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest",
                    str(output_path)
                ]
            else:
                # Replace audio entirely
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-i", str(tts_audio),
                    "-c:v", "copy",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest",
                    str(output_path)
                ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _cleanup_temp_files(self):
        """Remove temporary files from work directory."""
        patterns = ["tts_seg_*.mp3", "tts_seg_*.wav", "audio_raw.wav", "tts_en.wav"]
        for pattern in patterns:
            for f in self.cfg.work_dir.glob(pattern):
                try:
                    f.unlink()
                except Exception:
                    pass

    def _mux_replace_audio(self, video_path: Path, audio_path: Path, output_path: Path):
        """Legacy method - redirects to _mux_with_progress."""
        self._mux_with_progress(video_path, self.cfg.work_dir / "audio_raw.wav", audio_path, output_path)


def list_voices() -> List[Dict[str, str]]:
    """List available edge-tts voices."""
    import edge_tts
    
    async def get_voices():
        voices = await edge_tts.list_voices()
        return voices
    
    return asyncio.run(get_voices())


def print_voices(console: Console, filter_lang: str = "en"):
    """Print available voices in a formatted table."""
    voices = list_voices()
    
    # Filter by language
    if filter_lang:
        voices = [v for v in voices if v.get("Locale", "").startswith(filter_lang)]
    
    table = Table(title=f"Available Voices ({filter_lang or 'all'})", box=box.ROUNDED)
    table.add_column("Voice Name", style="cyan")
    table.add_column("Gender", style="green")
    table.add_column("Locale", style="yellow")
    
    for v in sorted(voices, key=lambda x: x.get("ShortName", "")):
        table.add_row(
            v.get("ShortName", ""),
            v.get("Gender", ""),
            v.get("Locale", "")
        )
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(voices)} voices[/dim]")


__all__ = ["Pipeline", "PipelineConfig", "list_voices", "print_voices"]
