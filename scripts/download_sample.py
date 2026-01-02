import os
import requests

def download_youtube_video(video_url, output_path):
    """
    Downloads a YouTube video using the provided URL and saves it to the specified output path.
    """
    # Use youtube-dl or pytube to download the video
    try:
        from pytube import YouTube
        yt = YouTube(video_url)
        stream = yt.streams.filter(only_audio=True).first()
        stream.download(output_path)
        print(f"Downloaded: {yt.title}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example usage
    video_url = input("Enter the YouTube video URL: ")
    output_path = os.path.join(os.getcwd(), "downloads")
    os.makedirs(output_path, exist_ok=True)
    download_youtube_video(video_url, output_path)