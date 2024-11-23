from football_analytics.library.utils import video_frames_generator

import pytest
import cv2
import numpy as np


@pytest.fixture
def sample_video(tmp_path):
    """
    Create sample video for testing purpose.
    :return: Video path
    """
    video_path = tmp_path / "sample_video.avi"

    height, width = 100, 100
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(video_path), fourcc, 1.0, (width, height))

    # 10 video frames
    for i in range(10):
        frame = (255 * (i % 2) * (i + 1) / 10) * np.ones((height, width, 3), dtype="uint8")
        out.write(frame.astype("uint8"))

    out.release()
    return str(video_path)


def test_video_frame_generator(sample_video):
    frames = list(video_frames_generator(sample_video))
    assert len(frames) == 10
    assert frames[0].shape == (100, 100, 3)
    assert frames[0].dtype == "uint8"


def test_video_file_not_found():
    with pytest.raises(FileNotFoundError):
        list(video_frames_generator("non_existent_file.mp4"))


def test_corrupted_video(tmp_path):
    corrupted_video_path = tmp_path / "corrupted_video.avi"
    with open(corrupted_video_path, "wb") as f:
        f.write(b"corrupted content")

    with pytest.raises(ValueError):
        list(video_frames_generator(str(corrupted_video_path)))


def test_empty_video(tmp_path):
    empty_video_path = tmp_path / "empty_video.avi"
    empty_video_path.touch()

    with pytest.raises(ValueError):
        list(video_frames_generator(str(empty_video_path)))
