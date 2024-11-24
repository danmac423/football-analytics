import supervision.detection
from core.utils import video_frames_generator, serialize_object

import pytest
import cv2
import numpy as np
from supervision.detection.core import Detections
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, Any, List


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


@pytest.fixture
def simple_data():
    return {
        "int": 42,
        "float": 3.14,
        "string": "hello",
        "none": None,
        "list": [1, 2, 3],
        "dict": {"key": "value", "number": 42},
        "tuple": (1, 2, 3),
        "numpy_array": np.array([1, 2, 3]),
    }


def test_serialize_simple_data(simple_data):
    serialized = serialize_object(simple_data)
    assert serialized == {
        "int": 42,
        "float": 3.14,
        "string": "hello",
        "none": None,
        "list": [1, 2, 3],
        "dict": {"key": "value", "number": 42},
        "tuple": [1, 2, 3],
        "numpy_array": [1, 2, 3],
    }


@dataclass
class SampleDetection:
    xyxy: np.ndarray
    mask: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None
    data: Dict[str, Union[np.ndarray, List]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def complex_object():
    """
    Fikstura z bardziej złożonym obiektem dataclass.
    """
    # Tworzenie obiektu DetectionResult
    detection = SampleDetection(
        xyxy=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
        mask=np.random.randint(0, 2, (2, 100, 100)),  # Maski binarne
        confidence=np.array([0.95, 0.89]),
        class_id=np.array([1, 2]),
        tracker_id=np.array([101, 102]),
        data={"additional_info": np.array([5, 10])},
        metadata={"source": "camera_1", "timestamp": 1234567890},
    )

    return detection


def test_serialize_complex_object(complex_object):
    serialized = serialize_object(complex_object)
    assert serialized == {
        "xyxy": [[10, 20, 30, 40], [50, 60, 70, 80]],
        "mask": complex_object.mask.tolist(),
        "confidence": complex_object.confidence.tolist(),
        "class_id": complex_object.class_id.tolist(),
        "tracker_id": complex_object.tracker_id.tolist(),
        "data": {"additional_info": [5, 10]},
        "metadata": {"source": "camera_1", "timestamp": 1234567890},
    }
