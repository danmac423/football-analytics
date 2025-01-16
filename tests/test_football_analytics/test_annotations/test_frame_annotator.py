from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from football_analytics.annotations.frame_annotator import FrameAnnotator
from services.ball_inference.grpc_files import ball_inference_pb2
from services.keypoints_detection.grpc_files import keypoints_detection_pb2
from services.player_inference.grpc_files import player_inference_pb2


@patch("football_analytics.annotations.frame_annotator.to_supervision")
@patch("football_analytics.annotations.frame_annotator.sv.EllipseAnnotator")
@patch("football_analytics.annotations.frame_annotator.sv.TriangleAnnotator")
@patch("football_analytics.annotations.frame_annotator.sv.VertexAnnotator")
def test_annotate_frame(mock_vertex, mock_triangle, mock_ellipse, mock_to_supervision):
    mock_ellipse_instance = MagicMock()
    mock_triangle_instance = MagicMock()
    mock_vertex_instance = MagicMock()
    mock_ellipse.return_value = mock_ellipse_instance
    mock_triangle.return_value = mock_triangle_instance
    mock_vertex.return_value = mock_vertex_instance

    mock_to_supervision.side_effect = lambda response, frame: MagicMock(
        xyxy=np.array([[10, 10, 20, 20]]),
        tracker_id=np.array([1]),
    )

    mock_ellipse_instance.annotate.side_effect = lambda frame, detections: frame
    mock_triangle_instance.annotate.side_effect = lambda frame, detections: frame
    mock_vertex_instance.annotate.side_effect = lambda frame, keypoints: frame

    annotator = FrameAnnotator()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    player_response = player_inference_pb2.PlayerInferenceResponse()
    player_response.boxes.add(x1_n=0.1, y1_n=0.2, x2_n=0.3, y2_n=0.4, tracker_id=1)

    ball_response = ball_inference_pb2.BallInferenceResponse()
    ball_response.boxes.add(x1_n=0.5, y1_n=0.5, x2_n=0.6, y2_n=0.6, confidence=0.8)

    keypoints_response = keypoints_detection_pb2.KeypointsDetectionResponse()
    keypoints_response.keypoints.add(x=50, y=50, confidence=0.9)

    velocities = {1: 5.0}

    annotated_frame = annotator.annotate_frame(frame, player_response, ball_response, velocities)

    assert annotated_frame is not None
    mock_ellipse_instance.annotate.assert_called_once()
    mock_triangle_instance.annotate.assert_called_once()
    mock_to_supervision.assert_called()


@patch("football_analytics.annotations.frame_annotator.to_supervision")
@patch("football_analytics.annotations.frame_annotator.sv.EllipseAnnotator")
@patch("football_analytics.annotations.frame_annotator.sv.TriangleAnnotator")
@patch("football_analytics.annotations.frame_annotator.sv.VertexAnnotator")
def test_annotate_frame_player_error(
    mock_vertex, mock_triangle, mock_ellipse, mock_to_supervision
):
    mock_ellipse_instance = MagicMock()
    mock_triangle_instance = MagicMock()
    mock_vertex_instance = MagicMock()

    mock_ellipse.return_value = mock_ellipse_instance
    mock_triangle.return_value = mock_triangle_instance
    mock_vertex.return_value = mock_vertex_instance

    def mock_to_supervision_side_effect(response, frame):
        if isinstance(response, player_inference_pb2.PlayerInferenceResponse):
            raise ValueError("Simulated player error")

    mock_to_supervision.side_effect = mock_to_supervision_side_effect

    annotator = FrameAnnotator()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    player_response = player_inference_pb2.PlayerInferenceResponse()
    player_response.boxes.add(x1_n=0.1, y1_n=0.2, x2_n=0.3, y2_n=0.4, tracker_id=1)

    ball_response = ball_inference_pb2.BallInferenceResponse()
    velocities = {}

    with pytest.raises(Exception, match="Error annotating players: Simulated player error"):
        annotator.annotate_frame(frame, player_response, ball_response, velocities)


@patch("football_analytics.annotations.frame_annotator.to_supervision")
@patch("football_analytics.annotations.frame_annotator.sv.EllipseAnnotator")
@patch("football_analytics.annotations.frame_annotator.sv.TriangleAnnotator")
@patch("football_analytics.annotations.frame_annotator.sv.VertexAnnotator")
def test_annotate_frame_ball_error(mock_vertex, mock_triangle, mock_ellipse, mock_to_supervision):
    mock_ellipse_instance = MagicMock()
    mock_triangle_instance = MagicMock()
    mock_vertex_instance = MagicMock()

    mock_ellipse.return_value = mock_ellipse_instance
    mock_triangle.return_value = mock_triangle_instance
    mock_vertex.return_value = mock_vertex_instance

    def mock_to_supervision_side_effect(response, frame):
        if isinstance(response, ball_inference_pb2.BallInferenceResponse):
            raise ValueError("Simulated ball error")

    mock_to_supervision.side_effect = mock_to_supervision_side_effect

    annotator = FrameAnnotator()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    player_response = None

    ball_response = ball_inference_pb2.BallInferenceResponse()
    ball_response.boxes.add(x1_n=0.5, y1_n=0.5, x2_n=0.6, y2_n=0.6, confidence=0.8)

    velocities = {}

    with pytest.raises(Exception, match="Error annotating ball: Simulated ball error"):
        annotator.annotate_frame(frame, player_response, ball_response, velocities)


@patch("football_analytics.annotations.frame_annotator.to_supervision")
@patch("football_analytics.annotations.frame_annotator.generate_radar")
def test_generate_radar(mock_generate_radar, mock_to_supervision):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_to_supervision.side_effect = lambda response, frame: MagicMock(
        xyxy=np.array([[10, 10, 20, 20]])
    )
    mock_generate_radar.return_value = frame

    player_response = player_inference_pb2.PlayerInferenceResponse()
    ball_response = ball_inference_pb2.BallInferenceResponse()
    keypoints_response = None

    radar_frame = FrameAnnotator.generate_radar(
        frame, player_response, ball_response, keypoints_response
    )

    assert np.array_equal(radar_frame, frame)

    player_response = None
    ball_response = None
    keypoints_response = keypoints_detection_pb2.KeypointsDetectionResponse()
    keypoints_response.keypoints.add(x=50, y=50, confidence=0.9)

    radar_frame = FrameAnnotator.generate_radar(
        frame, player_response, ball_response, keypoints_response
    )

    assert mock_generate_radar.called
    assert radar_frame is not None

    player_response = player_inference_pb2.PlayerInferenceResponse()
    player_response.boxes.add(x1_n=0.1, y1_n=0.2, x2_n=0.3, y2_n=0.4, tracker_id=1)

    ball_response = ball_inference_pb2.BallInferenceResponse()
    ball_response.boxes.add(x1_n=0.5, y1_n=0.5, x2_n=0.6, y2_n=0.6, confidence=0.8)

    keypoints_response = keypoints_detection_pb2.KeypointsDetectionResponse()
    keypoints_response.keypoints.add(x=50, y=50, confidence=0.9)

    radar_frame = FrameAnnotator.generate_radar(
        frame, player_response, ball_response, keypoints_response
    )

    assert radar_frame is not None
    mock_to_supervision.assert_called()
    mock_generate_radar.assert_called()
