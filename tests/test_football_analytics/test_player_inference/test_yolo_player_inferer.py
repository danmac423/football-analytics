from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv

from football_analytics.player_inference.yolo_players_inferer import (
    YOLOPlayerInferer,
)
from services.player_inference.grpc_files import player_inference_pb2


@pytest.fixture
def mock_inferer():
    with (
        patch("football_analytics.player_inference.yolo_players_inferer.YOLO") as mock_yolo,
        patch(
            "football_analytics.player_inference.yolo_players_inferer.sv.ByteTrack"
        ) as mock_tracker,
    ):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_tracker.return_value = MagicMock()
        inferer = YOLOPlayerInferer()
        return inferer


def test_load_model(mock_inferer):
    assert mock_inferer.model is not None


def test_initialize_tracker(mock_inferer):
    assert mock_inferer.tracker is not None


def test_reset_tracker(mock_inferer):
    mock_inferer.tracker.reset = MagicMock()
    mock_inferer.reset_tracker()
    mock_inferer.tracker.reset.assert_called_once()


def test_normalize_box(mock_inferer):
    box = [100, 200, 300, 400]
    width, height = 800, 600
    normalized_box = mock_inferer._normalize_box(box, width, height)

    assert np.isclose(normalized_box[0], 0.125)
    assert np.isclose(normalized_box[1], 0.3333333333333333)
    assert np.isclose(normalized_box[2], 0.375)
    assert np.isclose(normalized_box[3], 0.6666666666666666)


@patch("football_analytics.player_inference.yolo_players_inferer.sv.Detections.from_ultralytics")
def test_infer_players(mock_detections, mock_inferer):
    mock_detections.return_value = MagicMock(
        xyxy=[[100, 200, 300, 400]],
        confidence=[0.9],
        class_id=[0],
        tracker_id=[1],
    )
    mock_inferer.model.return_value = [MagicMock(names={0: "player"})]
    mock_inferer.tracker.update_with_detections.return_value = mock_detections.return_value

    frame_image = np.zeros((600, 800, 3), dtype=np.uint8)

    response = mock_inferer.infer_players(frame_image)

    assert isinstance(response, player_inference_pb2.PlayerInferenceResponse)
    assert len(response.boxes) == 1
    box = response.boxes[0]
    assert np.isclose(box.x1_n, 0.125)
    assert np.isclose(box.y1_n, 0.3333333333333333)
    assert np.isclose(box.x2_n, 0.375)
    assert np.isclose(box.y2_n, 0.6666666666666666)
    assert np.isclose(box.confidence, 0.9)
    assert box.class_label == "player"
    assert box.tracker_id == 1


@patch("football_analytics.player_inference.yolo_players_inferer.sv.Detections.from_ultralytics")
def test_infer_players_no_detections(mock_detections, mock_inferer):
    """Testuje metodę inferencji, gdy YOLO nie zwraca żadnych wykryć."""
    # Przygotowanie mocka dla wyników YOLO
    mock_results = MagicMock()
    mock_results.names = {}

    # Poprawne zamockowanie obiektu Detections
    mock_detections.return_value = sv.Detections(
        xyxy=np.empty((0, 4)),  # Puste bounding boxy
        confidence=np.array([]),  # Puste confidence
        class_id=np.array([]),  # Puste ID klas
        tracker_id=np.array([]),  # Puste ID trackerów
    )
    mock_inferer.model.return_value = [mock_results]
    mock_inferer.tracker.update_with_detections.return_value = mock_detections.return_value

    # Przygotowanie obrazu
    frame_image = np.zeros((600, 800, 3), dtype=np.uint8)

    # Wywołanie metody
    response = mock_inferer.infer_players(frame_image)

    # Sprawdzanie wyników
    assert isinstance(response, player_inference_pb2.PlayerInferenceResponse)
    assert len(response.boxes) == 0, "Powinna zwrócić pustą listę, gdy brak wykryć"


def test_infer_players_invalid_input(mock_inferer):
    with pytest.raises(AttributeError):
        mock_inferer.infer_players(None)
