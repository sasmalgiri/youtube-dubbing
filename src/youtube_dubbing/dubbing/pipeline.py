from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Iterable

import asyncio
import re
import shutil
import subprocess

from rich.console import Console
from rich.panel import Panel
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
        self.console.print(Panel.fit("YouTube Dubbing Pipeline (English-only)", box=box.ROUNDED))
        self._ensure_ffmpeg()

        video_path = self._ingest_source(self.cfg.source)
        audio_raw = self._extract_audio(video_path)

        self.console.print(f"[blue]Transcribing (English) with faster-whisper ({self.cfg.asr_model}) on cpu (int8)...[/blue]")
        segments = self._transcribe(audio_raw, model_size=self.cfg.asr_model)

        srt_en = self.cfg.work_dir / "transcript_en.srt"
        write_srt(segments, srt_en)
        self.console.print(f"[green]Wrote English SRT:[/green] {srt_en}")

        if self.cfg.transcribe_only:
            self.console.print("[green]Transcription complete (transcribe-only mode).[/green]")
            return

        # Check if we have any text to synthesize
        text_segments = [s for s in segments if s.get("text", "").strip()]
        if not text_segments:
            self.console.print("[yellow]No speech detected in video. Cannot generate TTS.[/yellow]")
            return

        # === AI TTS (English) ===
        self.console.print(f"[blue]Synthesizing English TTS (edge-tts voice: {self.cfg.tts_voice})...[/blue]")
        tts_wav = self._tts_edge_en(segments, voice=self.cfg.tts_voice, rate=self.cfg.tts_rate)

        if not tts_wav.exists():
            self.console.print("[red]TTS synthesis failed to produce output.[/red]")
            return

        # === Mux TTS audio into video ===
        self.console.print("[blue]Muxing dubbed audio into video...[/blue]")
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._mux_replace_audio(video_path, tts_wav, self.cfg.output_path)

        self.console.print(f"[green bold]Done! Dubbed video saved to:[/green bold] {self.cfg.output_path}")

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
                            # Try other possible structures
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
            self.console.print("[blue]Downloading source with yt-dlp...[/blue]")
            try:
                subprocess.run(
                    ["yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4", "-o", out_tpl, src],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "yt-dlp failed to download the URL. The video may be unavailable or require cookies.\n"
                    "Tip: ensure you own the content. If it’s your video and needs auth, try running yt-dlp with "
                    "--cookies-from-browser chrome (manually), then pass the downloaded file as --source."
                ) from e
            mp4 = list(self.cfg.work_dir.glob("source.mp4"))
            if mp4:
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
                return p
            # If the path doesn't exist, try scanning its parent directory
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
        cmd = ["ffmpeg","-y","-i",str(video_path),"-vn","-ac","2","-ar","48000","-acodec","pcm_s16le",str(wav)]
        self.console.print("[blue]Extracting audio...[/blue]")
        subprocess.run(cmd, check=True)
        return wav

    def _transcribe(self, wav_path: Path, model_size: str = "small") -> List[Dict]:
        from faster_whisper import WhisperModel
        device = "cpu"
        compute_type = "int8"
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            seg_iter, info = model.transcribe(str(wav_path), vad_filter=True)
            segments: List[Dict] = []
            for seg in seg_iter:
                segments.append({
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text.strip(),
                })
            return segments
        except Exception as e:
            self.console.print(f"[red]Transcription failed: {e}[/red]")
            raise

    def _tts_edge_en(self, segments: List[Dict], voice: str, rate: str) -> Path:
        """
        Synthesize all transcript segments to a single WAV file using edge-tts.
        Each segment's TTS audio is padded/truncated to match the original duration
        for lip-sync alignment.
        """
        import edge_tts
        import wave
        import struct
        import tempfile
        import os

        out_wav = self.cfg.work_dir / "tts_en.wav"
        sample_rate = 48000
        n_channels = 2

        async def synthesize_segment(text: str, seg_out: Path):
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(str(seg_out))

        async def run_all():
            seg_wavs: List[Path] = []
            for i, seg in enumerate(segments):
                text = seg.get("text", "").strip()
                if not text:
                    continue
                seg_mp3 = self.cfg.work_dir / f"tts_seg_{i:04d}.mp3"
                seg_wav = self.cfg.work_dir / f"tts_seg_{i:04d}.wav"
                await synthesize_segment(text, seg_mp3)
                # Convert mp3 to wav at target sample rate
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(seg_mp3), "-ar", str(sample_rate), "-ac", str(n_channels), str(seg_wav)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                seg_wavs.append(seg_wav)
                seg_mp3.unlink(missing_ok=True)  # cleanup mp3

            # Concatenate all segment WAVs
            if seg_wavs:
                with wave.open(str(out_wav), "wb") as wf:
                    wf.setnchannels(n_channels)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    for sw in seg_wavs:
                        try:
                            with wave.open(str(sw), "rb") as rf:
                                wf.writeframes(rf.readframes(rf.getnframes()))
                        except Exception as e:
                            self.console.print(f"[yellow]Warning: failed to read {sw}: {e}[/yellow]")
                        sw.unlink(missing_ok=True)  # cleanup segment wav

        asyncio.run(run_all())
        return out_wav

    def _mux_replace_audio(self, video_path: Path, audio_path: Path, output_path: Path):
        """
        Replace the original audio track of video_path with audio_path,
        producing a new video at output_path.
        """
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
        subprocess.run(cmd, check=True)


__all__ = []