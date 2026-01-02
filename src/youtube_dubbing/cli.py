from __future__ import annotations
import argparse
import sys
from pathlib import Path
from rich.console import Console
from youtube_dubbing.dubbing.pipeline_v2 import Pipeline, PipelineConfig, print_voices

console = Console()

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dub",
        description="YouTube Dubbing Pipeline v2 - AI-powered video dubbing with time-aligned TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dub --source video.mp4 --output dubbed.mp4
  dub --source video.mp4 --tts-voice en-US-GuyNeural
  dub --source video.mp4 --mix-original --original-volume 0.2
  dub --list-voices
  dub --list-voices --language de
        """
    )
    
    # Source & output
    p.add_argument("--source", help="Local video file or YouTube URL")
    p.add_argument("--work-dir", default="work", help="Working directory (default: work)")
    p.add_argument("--output", default="output/video_dubbed.mp4", help="Output file path")
    
    # ASR options
    p.add_argument("--asr-model", default="small",
                   choices=["tiny", "small", "medium", "large-v3"],
                   help="Whisper model size (default: small)")
    p.add_argument("--transcribe-only", action="store_true",
                   help="Only transcribe, don't generate TTS")
    
    # TTS options
    p.add_argument("--tts-voice", default="en-US-AriaNeural",
                   help="Edge-TTS voice (default: en-US-AriaNeural)")
    p.add_argument("--tts-rate", default="+0%",
                   help="Speech rate adjustment (e.g., +10%%, -5%%)")
    p.add_argument("--no-time-align", action="store_true",
                   help="Disable time alignment (just concatenate TTS)")
    
    # Audio mixing
    p.add_argument("--mix-original", action="store_true",
                   help="Blend original audio with TTS")
    p.add_argument("--original-volume", type=float, default=0.15,
                   help="Original audio volume when mixing (0.0-1.0, default: 0.15)")
    
    # Voice listing
    p.add_argument("--list-voices", action="store_true",
                   help="List available TTS voices and exit")
    p.add_argument("--language", default="en",
                   help="Language filter for --list-voices (default: en)")
    
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Handle --list-voices
    if args.list_voices:
        print_voices(console, args.language)
        return

    # Require --source for normal operation
    if not args.source:
        console.print("[red]Error: --source is required[/red]")
        console.print("Use --help for usage information")
        sys.exit(1)

    cfg = PipelineConfig(
        source=args.source,
        work_dir=Path(args.work_dir),
        output_path=Path(args.output),
        asr_model=args.asr_model,
        transcribe_only=args.transcribe_only,
        tts_voice=args.tts_voice,
        tts_rate=args.tts_rate,
        time_aligned=not args.no_time_align,
        mix_original=args.mix_original,
        original_volume=args.original_volume,
    )

    pipeline = Pipeline(cfg, console=console)
    pipeline.run()

if __name__ == "__main__":
    main()