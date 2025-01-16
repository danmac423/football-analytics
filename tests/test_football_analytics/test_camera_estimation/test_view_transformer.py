from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from football_analytics.camera_estimation.view_transformer import ViewTransformer
from services.keypoints_detection.grpc_files import keypoints_detection_pb2


def test_view_transformer_initialization():
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])

    transformer = ViewTransformer(source, target)
    assert transformer.metrics is not None
    assert transformer.metrics.shape == (3, 3)


def test_view_transformer_initialization_empty_matrices():
    source = np.array([])
    target = np.array([])

    with pytest.raises(cv2.error):
        ViewTransformer(source, target)


def test_transform_points():
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    points = np.array([[0.5, 0.5], [0.75, 0.75]])

    transformer = ViewTransformer(source, target)

    transformed_points = transformer.transform_points(points)

    assert transformed_points.shape == (2, 2)
    assert np.all(transformed_points >= 0)


def test_transform_points_empty():
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    points = np.array([])

    transformer = ViewTransformer(source, target)

    transformed_points = transformer.transform_points(points)

    assert transformed_points.size == 0


@patch("football_analytics.camera_estimation.view_transformer.to_supervision")
@patch("football_analytics.camera_estimation.view_transformer.FootballPitchConfiguration")
def test_get_view_transformer(mock_config, mock_to_supervision):
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    keypoints_response = keypoints_detection_pb2.KeypointsDetectionResponse()
    for x, y, conf in [(100, 200, 0.8), (300, 400, 0.9), (500, 600, 0.85), (700, 800, 0.95)]:
        keypoint = keypoints_response.keypoints.add()
        keypoint.x = x
        keypoint.y = y
        keypoint.confidence = conf

    mock_keypoints = MagicMock()
    mock_keypoints.confidence = np.array([0.8, 0.9, 0.85, 0.95])
    mock_keypoints.xy = np.array([[[100, 200], [300, 400], [500, 600], [700, 800]]])
    mock_to_supervision.return_value = mock_keypoints

    mock_config.return_value.vertices = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
    )

    transformer = ViewTransformer.get_view_transformer(frame, keypoints_response)

    assert isinstance(transformer, ViewTransformer)
    assert transformer.metrics is not None
    assert transformer.metrics.shape == (3, 3)


def test_get_view_transformer_no_keypoints():
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    keypoints_response = keypoints_detection_pb2.KeypointsDetectionResponse()

    with pytest.raises(
        ValueError, match="Keypoints not available for initializing ViewTransformer."
    ):
        ViewTransformer.get_view_transformer(frame, keypoints_response)
