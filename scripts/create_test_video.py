"""
Generate a test video with speech for testing the dubbing pipeline.
Uses edge-tts to create audio and FFmpeg to create a simple video.
"""
import asyncio
import subprocess
import shutil
import os
import sys
from pathlib import Path


def ensure_ffmpeg():
    """Find and add FFmpeg to PATH if needed."""
    if shutil.which("ffmpeg"):
        return True
    
    if sys.platform == "win32":
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
                                print(f"Found FFmpeg: {bin_path}")
                                return True
    return False


async def create_test_video():
    import edge_tts
    
    work_dir = Path("work")
    work_dir.mkdir(exist_ok=True)
    
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    
    # Test script with natural pauses
    script = """
    Hello and welcome to this test video.
    This is a demonstration of the YouTube dubbing pipeline.
    The pipeline uses faster-whisper for speech recognition.
    And edge TTS for text to speech synthesis.
    With time-aligned audio, the dubbed speech matches the original timing.
    Thank you for watching this demonstration.
    """
    
    audio_path = work_dir / "test_speech.mp3"
    wav_path = work_dir / "test_speech.wav"
    video_path = input_dir / "test_video.mp4"
    
    print("Generating speech with edge-tts...")
    communicate = edge_tts.Communicate(script.strip(), "en-US-GuyNeural", rate="-10%")
    await communicate.save(str(audio_path))
    
    print("Converting to WAV...")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ar", "48000", "-ac", "2",
        str(wav_path)
    ], check=True, capture_output=True)
    
    # Get audio duration
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(wav_path)
    ], capture_output=True, text=True)
    duration = float(result.stdout.strip())
    
    print(f"Creating {duration:.1f}s test video...")
    
    # Create a simple video with colored background and text
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x1a1a2e:s=1280x720:d={duration}",
        "-i", str(wav_path),
        "-vf", (
            "drawtext=text='Test Video for Dubbing Pipeline':"
            "fontsize=48:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2-50,"
            "drawtext=text='Original Audio':"
            "fontsize=32:fontcolor=0x00ff88:x=(w-text_w)/2:y=(h-text_h)/2+30"
        ),
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        "-shortest",
        str(video_path)
    ], check=True)
    
    # Cleanup
    audio_path.unlink(missing_ok=True)
    wav_path.unlink(missing_ok=True)
    
    print(f"\n✓ Test video created: {video_path}")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"\nRun the pipeline with:")
    print(f'  .venv\\Scripts\\python.exe -m youtube_dubbing.cli --source "{video_path}"')

if __name__ == "__main__":
    if not ensure_ffmpeg():
        print("Error: FFmpeg not found. Install via: winget install Gyan.FFmpeg")
        sys.exit(1)
    asyncio.run(create_test_video())
