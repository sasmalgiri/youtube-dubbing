import sys
import os
from moviepy.editor import VideoFileClip

def extract_audio(video_path, output_audio_path):
    if not os.path.exists(video_path):
        print(f"Error: The video file {video_path} does not exist.")
        return

    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_audio_path)
        print(f"Audio extracted successfully to {output_audio_path}")
    except Exception as e:
        print(f"An error occurred while extracting audio: {e}")
    finally:
        video.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_audio.py <video_path> <output_audio_path>")
    else:
        video_path = sys.argv[1]
        output_audio_path = sys.argv[2]
        extract_audio(video_path, output_audio_path)