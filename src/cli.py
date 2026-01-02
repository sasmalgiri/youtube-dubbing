import argparse
from src.dubbing.pipeline import DubbingPipeline

def main():
    parser = argparse.ArgumentParser(description="YouTube Video Dubbing Application")
    parser.add_argument('video_url', type=str, help='URL of the YouTube video to be dubbed')
    parser.add_argument('target_language', type=str, help='Language code for the target dubbing language')
    parser.add_argument('--output', type=str, default='dubbed_video.mp4', help='Output file name for the dubbed video')
    
    args = parser.parse_args()

    pipeline = DubbingPipeline()
    pipeline.process_video(args.video_url, args.target_language, args.output)

if __name__ == "__main__":
    main()