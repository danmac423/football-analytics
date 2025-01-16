from unittest.mock import MagicMock, patch

import numpy as np

from football_analytics.ball_inference.ball_tracker import BallTracker
from football_analytics.ball_inference.yolo_ball_inferer import YOLOBallInferer
from services.ball_inference.grpc_files import ball_inference_pb2


@patch("football_analytics.ball_inference.yolo_ball_inferer.YOLO")
def test_yolo_ball_inferer_initialization(mock_yolo):
    """Testuje inicjalizację YOLOBallInferer."""
    mock_yolo.return_value = MagicMock()
    inferer = YOLOBallInferer()

    assert isinstance(inferer.tracker, BallTracker), "Tracker powinien być instancją BallTracker."
    assert inferer.model is not None, "Model YOLO powinien być zainicjalizowany."


def test_yolo_ball_inferer_reset_tracker():
    """Testuje resetowanie trackera."""
    inferer = YOLOBallInferer()
    inferer.tracker = MagicMock()
    inferer.reset_tracker()

    inferer.tracker.reset.assert_called_once()


@patch("football_analytics.ball_inference.yolo_ball_inferer.sv.Detections")
@patch("football_analytics.ball_inference.yolo_ball_inferer.YOLO")
def test_yolo_ball_inferer_infer_ball(mock_yolo, mock_detections):
    """Testuje metodę infer_ball."""
    mock_model = MagicMock()
    mock_model.return_value = [MagicMock()]
    mock_yolo.return_value = mock_model
    mock_detections.from_ultralytics.return_value = MagicMock(
        xyxy=np.array([[10, 20, 30, 40]]), confidence=np.array([0.9])
    )

    frame_image = np.zeros((600, 800, 3), dtype=np.uint8)
    inferer = YOLOBallInferer()
    inferer.model = mock_model
    inferer._create_inference_slicer = MagicMock()
    inferer._perform_inference = MagicMock(
        return_value=mock_detections.from_ultralytics.return_value
    )
    inferer._format_detections = MagicMock(return_value=[])

    response = inferer.infer_ball(frame_image)

    assert isinstance(response, ball_inference_pb2.BallInferenceResponse), (
        "Powinien zwracać BallInferenceResponse."
    )
    inferer._perform_inference.assert_called_once()
    inferer._format_detections.assert_called_once()


@patch("football_analytics.ball_inference.yolo_ball_inferer.sv.InferenceSlicer")
def test_yolo_ball_inferer_create_inference_slicer(mock_inference_slicer):
    """Testuje tworzenie slicera."""
    inferer = YOLOBallInferer()

    slicer = inferer._create_inference_slicer(800, 600)
    assert slicer is not None, "Slicer powinien być poprawnie utworzony."


@patch("football_analytics.ball_inference.yolo_ball_inferer.sv.Detections")
@patch("football_analytics.ball_inference.yolo_ball_inferer.YOLO")
def test_yolo_ball_inferer_perform_inference(mock_yolo, mock_detections):
    """Testuje wykonywanie inferencji."""
    mock_model = MagicMock()
    mock_model.return_value = [MagicMock()]
    mock_yolo.return_value = mock_model
    mock_detections.from_ultralytics.return_value = MagicMock(
        xyxy=np.array([[10, 20, 30, 40]]), confidence=np.array([0.9])
    )

    frame_image = np.zeros((600, 800, 3), dtype=np.uint8)
    slicer = MagicMock()
    tracker = MagicMock()

    inferer = YOLOBallInferer()
    detections = inferer._perform_inference(frame_image, slicer, tracker)

    slicer.assert_called_once_with(frame_image)
    tracker.update.assert_called_once()
    assert detections is not None, "Wynik inferencji nie powinien być pusty."


def test_yolo_ball_inferer_format_detections():
    """Testuje formatowanie wyników."""
    detections = MagicMock(xyxy=np.array([[10, 20, 30, 40]]), confidence=np.array([0.9]))
    inferer = YOLOBallInferer()

    boxes = inferer._format_detections(detections, width=800, height=600)
    assert len(boxes) == 1, "Powinna być dokładnie jedna wykryta ramka."
    assert isinstance(boxes[0], ball_inference_pb2.BoundingBox), (
        "Każda ramka powinna być instancją BoundingBox."
    )
