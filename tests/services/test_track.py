from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from services.track.app import app
from supervision.detection.core import Detections
import numpy as np

client = TestClient(app)


@pytest.fixture
def mock_yolo_and_detections():
    with (
        patch("services.track.routes.infer.model") as mock_model,
        patch(
            "services.track.routes.infer.sv.Detections.from_ultralytics"
        ) as mock_detections_method,
    ):
        mock_model.predict.return_value = ["mock_result"]

        mock_detections_method.return_value = Detections(
            xyxy=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
            confidence=np.array([0.95, 0.89]),
            class_id=np.array([1, 2]),
        )

        yield mock_model, mock_detections_method


def test_infer_endpoint_with_mocked_yolo_and_detections(mock_yolo_and_detections):
    test_image = b"\x00" * 100
    response = client.post(
        "/infer",
        files={"file": ("test_image.jpg", test_image, "image/jpeg")},
        data={"height": 10, "width": 10, "channels": 1},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["xyxy"] == [[10, 20, 30, 40], [50, 60, 70, 80]]
    assert data["confidence"] == [0.95, 0.89]
    assert data["class_id"] == [1, 2]
