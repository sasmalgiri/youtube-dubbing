from typing import List, Dict, Any

class VideoDubbingConfig:
    def __init__(self, video_url: str, target_language: str, output_format: str):
        self.video_url = video_url
        self.target_language = target_language
        self.output_format = output_format

class Subtitle:
    def __init__(self, start_time: float, end_time: float, text: str):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text

class DubbedAudioTrack:
    def __init__(self, language: str, audio_file: str, subtitles: List[Subtitle]):
        self.language = language
        self.audio_file = audio_file
        self.subtitles = subtitles

class DubbingResult:
    def __init__(self, video_url: str, dubbed_tracks: List[DubbedAudioTrack]):
        self.video_url = video_url
        self.dubbed_tracks = dubbed_tracks

class DubbingError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message