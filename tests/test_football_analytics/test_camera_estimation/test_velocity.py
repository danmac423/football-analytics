from unittest.mock import MagicMock

import numpy as np
import pytest

from football_analytics.camera_estimation.velocity import VelocityEstimator
from football_analytics.camera_estimation.view_transformer import ViewTransformer
from services.player_inference.grpc_files import player_inference_pb2


def test_estimate_velocity_no_previous_position():
    view_transformer = MagicMock(spec=ViewTransformer)
    view_transformer.transform_points.return_value = np.array([[10, 20]])

    estimator = VelocityEstimator()
    velocity = estimator.estimate_velocity(view_transformer, (5, 10), tracker_id=1, delta_time=0.5)

    assert velocity == 0
    assert estimator.previous_positions[1] == (5, 10)


def test_estimate_velocity_with_previous_position():
    view_transformer = MagicMock(spec=ViewTransformer)
    view_transformer.transform_points.side_effect = [
        np.array([[10, 20]]),  # current_position
        np.array([[5, 15]]),  # previous_position
    ]

    estimator = VelocityEstimator()
    estimator.previous_positions[1] = (5, 10)

    velocity = estimator.estimate_velocity(
        view_transformer, (10, 20), tracker_id=1, delta_time=0.5
    )

    assert velocity > 0
    assert estimator.previous_positions[1] == (10, 20)


def test_estimate_velocity_zero_delta_time():
    view_transformer = MagicMock(spec=ViewTransformer)
    view_transformer.transform_points.return_value = np.array([[10, 20]])

    estimator = VelocityEstimator()
    velocity = estimator.estimate_velocity(view_transformer, (5, 10), tracker_id=1, delta_time=0)

    assert velocity == 0


def test_estimate_velocities_empty_response():
    view_transformer = MagicMock(spec=ViewTransformer)
    estimator = VelocityEstimator()

    player_response = player_inference_pb2.PlayerInferenceResponse()
    frame_ndarray = np.zeros((600, 800, 3), dtype=np.uint8)
    delta_time = 0.5

    velocities = estimator.estimate_velocities(
        view_transformer, player_response, frame_ndarray, delta_time
    )

    assert velocities == {}


def test_estimate_velocities_with_detections():
    view_transformer = MagicMock(spec=ViewTransformer)
    view_transformer.transform_points.side_effect = [
        np.array([[10, 20]]),  # current_position
        np.array([[5, 15]]),  # previous_position
    ]

    estimator = VelocityEstimator()
    estimator.previous_positions[1] = (5, 10)

    player_response = player_inference_pb2.PlayerInferenceResponse()
    player_response.boxes.extend(
        [
            player_inference_pb2.BoundingBox(
                x1_n=0.1, y1_n=0.2, x2_n=0.3, y2_n=0.4, tracker_id=1, class_label="player"
            ),
        ]
    )

    frame_ndarray = np.zeros((600, 800, 3), dtype=np.uint8)
    delta_time = 0.5

    velocities = estimator.estimate_velocities(
        view_transformer, player_response, frame_ndarray, delta_time
    )

    assert 1 in velocities
    assert velocities[1] > 0


def test_estimate_velocities_with_error():
    view_transformer = MagicMock(spec=ViewTransformer)
    view_transformer.transform_points.side_effect = ValueError("Error in transform_points")

    estimator = VelocityEstimator()

    player_response = player_inference_pb2.PlayerInferenceResponse()
    player_response.boxes.extend(
        [
            player_inference_pb2.BoundingBox(
                x1_n=0.1, y1_n=0.2, x2_n=0.3, y2_n=0.4, tracker_id=1, class_label="player"
            )
        ]
    )

    frame_ndarray = np.zeros((600, 800, 3), dtype=np.uint8)
    delta_time = 0.5

    with pytest.raises(Exception, match="Error annotating players:"):
        estimator.estimate_velocities(view_transformer, player_response, frame_ndarray, delta_time)
