def extract_video_frames(video_path, output_folder):
    import cv2
    import os

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    video_capture.release()

def get_video_duration(video_path):
    import cv2

    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    video_capture.release()
    return duration

def resize_video(video_path, output_path, width, height):
    import cv2

    video_capture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (width, height))
        out.write(resized_frame)

    video_capture.release()
    out.release()