from typing import List, Tuple

def align_audio_segments(audio_segments: List[Tuple[float, float]], video_duration: float) -> List[Tuple[float, float]]:
    aligned_segments = []
    current_time = 0.0

    for start, end in audio_segments:
        if start < current_time:
            start = current_time
        if end > video_duration:
            end = video_duration
        
        aligned_segments.append((start, end))
        current_time = end

    return aligned_segments

def main():
    # Example usage
    audio_segments = [(0.0, 2.0), (2.5, 4.0), (4.5, 6.0)]
    video_duration = 10.0
    aligned_segments = align_audio_segments(audio_segments, video_duration)
    print("Aligned Audio Segments:", aligned_segments)

if __name__ == "__main__":
    main()