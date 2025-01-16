import numpy as np
import pytest
import supervision as sv

from football_analytics.utils.model import to_supervision
from services.ball_inference.grpc_files import ball_inference_pb2
from services.keypoints_detection.grpc_files import keypoints_detection_pb2
from services.player_inference.grpc_files import player_inference_pb2


def test_to_supervision_ball_empty():
    ball_response = ball_inference_pb2.BallInferenceResponse(boxes=[])
    frame_ndarray = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = to_supervision(ball_response, frame_ndarray)

    assert isinstance(detections, sv.Detections)
    assert detections.xyxy.shape == (0, 4)


def test_to_supervision_ball_valid():
    ball_response = ball_inference_pb2.BallInferenceResponse(
        boxes=[
            ball_inference_pb2.BoundingBox(
                x1_n=0.1, y1_n=0.1, x2_n=0.2, y2_n=0.2, confidence=0.9, class_label="ball"
            )
        ]
    )
    frame_ndarray = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = to_supervision(ball_response, frame_ndarray)

    assert isinstance(detections, sv.Detections)
    assert detections.xyxy.shape == (1, 4)
    assert detections.confidence[0] == np.float32(0.9)


def test_to_supervision_ball_value_error():
    ball_response = ball_inference_pb2.BallInferenceResponse(
        boxes=[
            ball_inference_pb2.BoundingBox(
                x1_n=1.5,
                y1_n=1.1,
                x2_n=1.2,
                y2_n=1.2,
                confidence=0.9,
                class_label="ball",
            )
        ]
    )
    frame_ndarray = np.zeros((720, 1280, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        to_supervision(ball_response, frame_ndarray)


def test_to_supervision_keypoints_empty():
    keypoints_response = keypoints_detection_pb2.KeypointsDetectionResponse(keypoints=[])
    frame_ndarray = np.zeros((720, 1280, 3), dtype=np.uint8)

    keypoints = to_supervision(keypoints_response, frame_ndarray)

    assert isinstance(keypoints, sv.KeyPoints)
    assert keypoints.xy.shape == (0, 0, 2)


def test_to_supervision_keypoints_valid():
    keypoints_response = keypoints_detection_pb2.KeypointsDetectionResponse(
        keypoints=[
            keypoints_detection_pb2.Keypoint(x=100, y=200, confidence=0.8),
            keypoints_detection_pb2.Keypoint(x=300, y=400, confidence=0.9),
        ]
    )
    frame_ndarray = np.zeros((720, 1280, 3), dtype=np.uint8)

    keypoints = to_supervision(keypoints_response, frame_ndarray)

    assert isinstance(keypoints, sv.KeyPoints)
    assert keypoints.xy.shape == (1, 2, 2)
    assert keypoints.confidence.shape == (1, 2)
    assert keypoints.confidence[0][0] == np.float32(0.8)
    assert keypoints.confidence[0][1] == np.float32(0.9)


def test_to_supervision_player_empty():
    player_response = player_inference_pb2.PlayerInferenceResponse(boxes=[])
    frame_ndarray = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = to_supervision(player_response, frame_ndarray)

    assert isinstance(detections, sv.Detections)
    assert detections.xyxy.shape == (0, 4)


def test_to_supervision_player_valid():
    player_response = player_inference_pb2.PlayerInferenceResponse(
        boxes=[
            player_inference_pb2.BoundingBox(
                x1_n=0.1,
                y1_n=0.1,
                x2_n=0.3,
                y2_n=0.3,
                confidence=0.85,
                class_label="player",
                tracker_id=1,
            )
        ]
    )
    frame_ndarray = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = to_supervision(player_response, frame_ndarray)

    assert isinstance(detections, sv.Detections)
    assert detections.xyxy.shape == (1, 4)
    assert detections.confidence[0] == np.float32(0.85)
    assert detections.tracker_id[0] == 1


def test_to_supervision_player_unknown_class_label():
    player_response = player_inference_pb2.PlayerInferenceResponse(
        boxes=[
            player_inference_pb2.BoundingBox(
                x1_n=0.1,
                y1_n=0.1,
                x2_n=0.3,
                y2_n=0.3,
                confidence=0.85,
                class_label="unknown",
                tracker_id=1,
            )
        ]
    )
    frame_ndarray = np.zeros((720, 1280, 3), dtype=np.uint8)

    with pytest.raises(KeyError, match="unknown"):
        to_supervision(player_response, frame_ndarray)


def test_to_supervision_generic_exception():
    invalid_response = None
    frame_ndarray = np.zeros((720, 1280, 3), dtype=np.uint8)

    with pytest.raises(Exception):
        to_supervision(invalid_response, frame_ndarray)
