from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from football_analytics.player_inference.team_processor import TeamAssignmentProcessor
from services.player_inference.grpc_files import player_inference_pb2

PLAYER_ID = 1


@pytest.fixture
def mock_processor():
    with (
        patch(
            "football_analytics.player_inference.team_processor.SiglipVisionModel"
        ) as mock_model,
        patch(
            "football_analytics.player_inference.team_processor.AutoProcessor"
        ) as mock_auto_processor,
        patch("football_analytics.player_inference.team_processor.umap.UMAP") as mock_umap,
        patch("football_analytics.player_inference.team_processor.KMeans") as mock_kmeans,
    ):
        mock_model.from_pretrained.return_value = MagicMock()
        mock_auto_processor.from_pretrained.return_value = MagicMock()
        mock_umap.return_value = MagicMock()
        mock_kmeans.return_value = MagicMock()
        processor = TeamAssignmentProcessor()
        return processor


def test_processor_initialization(mock_processor):
    assert mock_processor.embedding_model is not None
    assert mock_processor.embedding_processor is not None
    assert isinstance(mock_processor.reducer, MagicMock)
    assert isinstance(mock_processor.clustering_model, MagicMock)


@patch("football_analytics.player_inference.team_processor.sv.cv2_to_pillow")
@patch("football_analytics.player_inference.team_processor.torch.no_grad")
def test_extract_features(mock_torch_no_grad, mock_cv2_to_pillow, mock_processor):
    crops = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(10)]

    mock_cv2_to_pillow.side_effect = lambda x: x

    mock_processor.embedding_processor.return_value = MagicMock(
        images=lambda images, return_tensors: {"mock_inputs": True}
    )

    mock_processor.embedding_model.return_value = MagicMock(
        last_hidden_state=torch.rand(10, 3, 224)
    )

    features = mock_processor.extract_features(crops, batch_size=5)

    assert features.shape[0] == 20
    assert features.shape[1] == 224

    assert mock_cv2_to_pillow.call_count == len(crops)


@patch("football_analytics.player_inference.team_processor.umap.UMAP")
@patch("football_analytics.player_inference.team_processor.KMeans")
def test_fit_clustering_model(mock_kmeans, mock_umap, mock_processor):
    features = np.random.rand(100, 128)

    mock_processor.reducer.fit_transform.return_value = np.random.rand(100, 3)
    mock_processor.clustering_model.fit = MagicMock()

    mock_processor.fit_clustering_model(features)

    mock_processor.reducer.fit_transform.assert_called_once_with(features)
    mock_processor.clustering_model.fit.assert_called_once_with(
        mock_processor.reducer.fit_transform.return_value
    )


@patch("football_analytics.player_inference.team_processor.umap.UMAP")
@patch("football_analytics.player_inference.team_processor.KMeans")
def test_predict_teams(mock_kmeans, mock_umap, mock_processor):
    features = np.random.rand(10, 128)

    mock_processor.reducer.transform.return_value = np.random.rand(10, 3)
    mock_processor.clustering_model.predict.return_value = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    predictions = mock_processor.predict_teams(features)

    assert len(predictions) == 10
    assert set(predictions) <= {0, 1}
    mock_processor.reducer.transform.assert_called_once_with(features)
    mock_processor.clustering_model.predict.assert_called_once_with(
        mock_processor.reducer.transform.return_value
    )


from supervision.detection.core import Detections


@patch("football_analytics.player_inference.team_processor.cv2.imdecode")
@patch("football_analytics.player_inference.team_processor.sv.crop_image")
@patch("football_analytics.player_inference.team_processor.to_supervision")
def test_collect_crops(mock_to_supervision, mock_crop_image, mock_imdecode, mock_processor):
    mock_frames = [MagicMock(content=b"mock_frame")] * 3
    mock_player_response = player_inference_pb2.PlayerInferenceResponse()
    mock_player_response.boxes.extend(
        [player_inference_pb2.BoundingBox(x1_n=0.1, y1_n=0.1, x2_n=0.3, y2_n=0.3)]
    )
    mock_player_responses = [mock_player_response] * 3

    detections = Detections(
        xyxy=np.array([[10, 10, 30, 30]]), confidence=np.array([0.9]), class_id=np.array([1])
    )
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)
    mock_to_supervision.return_value = detections

    mock_imdecode.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
    mock_crop_image.side_effect = lambda frame, box: np.ones((50, 50, 3), dtype=np.uint8)

    crops = mock_processor.collect_crops(mock_player_responses, mock_frames)

    assert len(crops) == 3
    assert mock_imdecode.call_count == len(mock_frames)
    assert mock_crop_image.call_count == len(mock_frames)
