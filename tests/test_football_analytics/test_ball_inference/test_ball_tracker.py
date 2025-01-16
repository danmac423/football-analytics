from collections import deque
from unittest.mock import MagicMock

import numpy as np
import supervision as sv

from football_analytics.ball_inference.ball_tracker import BallTracker


def test_ball_tracker_initialization():
    tracker = BallTracker(buffer_size=5)
    assert isinstance(tracker.buffer, deque)
    assert tracker.buffer.maxlen == 5


def test_ball_tracker_update_no_detections():
    tracker = BallTracker(buffer_size=5)
    mock_detections = MagicMock(spec=sv.Detections)
    mock_detections.get_anchors_coordinates.return_value = np.array([])

    updated_detections = tracker.update(mock_detections)
    assert updated_detections == mock_detections
    assert len(tracker.buffer) == 1


def test_ball_tracker_update_with_detections():
    tracker = BallTracker(buffer_size=5)
    mock_detections = MagicMock(spec=sv.Detections)
    mock_detections.get_anchors_coordinates.return_value = np.array([[10, 20], [30, 40]])
    mock_detections.__len__.return_value = 2

    updated_detections = tracker.update(mock_detections)
    assert len(tracker.buffer) == 1
    assert updated_detections is not None
    assert updated_detections == mock_detections[[0]]


def test_ball_tracker_update_with_multiple_updates():
    tracker = BallTracker(buffer_size=3)
    mock_detections = MagicMock(spec=sv.Detections)
    mock_detections.get_anchors_coordinates.side_effect = [
        np.array([[10, 20]]),
        np.array([[15, 25]]),
        np.array([[20, 30]]),
    ]
    mock_detections.__len__.return_value = 1

    for _ in range(3):
        tracker.update(mock_detections)

    assert len(tracker.buffer) == 3
    assert np.allclose(tracker.buffer[-1], np.array([[20, 30]]))


def test_ball_tracker_reset():
    tracker = BallTracker(buffer_size=5)
    mock_detections = MagicMock(spec=sv.Detections)
    mock_detections.get_anchors_coordinates.return_value = np.array([[10, 20]])
    mock_detections.__len__.return_value = 1

    tracker.update(mock_detections)
    assert len(tracker.buffer) == 1

    tracker.reset()
    assert len(tracker.buffer) == 0
