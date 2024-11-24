from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from services.track.app import app  # Import aplikacji FastAPI
from supervision.detection.core import Detections
import numpy as np

client = TestClient(app)


@pytest.fixture
def mock_yolo_and_detections():
    """
    Mockowanie modelu YOLO i funkcji sv.Detections.from_ultralytics.
    """
    with (
        patch("services.track.routes.infer.model") as mock_model,
        patch(
            "services.track.routes.infer.sv.Detections.from_ultralytics"
        ) as mock_detections_method,
    ):

        mock_model.predict.return_value = ["mock_result"]  # Wynik modelu YOLO

        mock_detections_method.return_value = Detections(
            xyxy=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
            confidence=np.array([0.95, 0.89]),
            class_id=np.array([1, 2]),
        )

        yield mock_model, mock_detections_method


def test_infer_endpoint_with_mocked_yolo_and_detections(mock_yolo_and_detections):
    """
    Test endpointu /infer z zamockowanym modelem YOLO i Detections.from_ultralytics.
    """
    test_image = b"\x00" * 100  # Przykładowy plik wejściowy
    response = client.post(
        "/infer",
        files={"file": ("test_image.jpg", test_image, "image/jpeg")},
    )

    assert response.status_code == 200

    # Sprawdzenie odpowiedzi JSON
    data = response.json()
    assert data["xyxy"] == [[10, 20, 30, 40], [50, 60, 70, 80]]
    assert data["confidence"] == [0.95, 0.89]
    assert data["class_id"] == [1, 2]
