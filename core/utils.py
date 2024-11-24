import cv2
import os


def video_frames_generator(video_path):
    """
    Generator which returns next video frames

    Args:
        video_path (string): path to video file

    Yield:
        Video frame in OpenCV fromat
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Plik wideo nie istnieje: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()
