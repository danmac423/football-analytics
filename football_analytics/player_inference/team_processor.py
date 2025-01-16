"""Module responsible for assigning players to teams."""

import cv2
import numpy as np
import supervision as sv
import torch
import umap
from more_itertools import chunked
from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel

from config import DEVICE, PLAYER_ID
from football_analytics.utils.model import to_supervision
from services.player_inference.grpc_files import player_inference_pb2


class TeamAssignmentProcessor:
    """
    Class responsible for assigning players to teams.

    Attributes:
        embedding_model (SiglipVisionModel): The embedding model.
        embedding_processor (AutoProcessor): The embedding processor.
        reducer (umap.UMAP): The UMAP reducer.
        clustering_model (KMeans): The clustering model.

    Args:
        embedding_model_path (str): The path to the embedding model.
        n_clusters (int): The number of clusters.
    """

    def __init__(
        self, embedding_model_path: str = "google/siglip-base-patch16-224", n_clusters: int = 2
    ):
        self.embedding_model = SiglipVisionModel.from_pretrained(embedding_model_path).to(DEVICE)
        self.embedding_processor = AutoProcessor.from_pretrained(embedding_model_path)
        self.reducer = umap.UMAP(n_components=3)
        self.clustering_model = KMeans(n_clusters=n_clusters)

    def extract_features(self, crops: list, batch_size: int = 32) -> np.ndarray:
        """Extracts features from image crops using the embedding model.

        Args:
            crops (list): List of cropped player images.
            batch_size (int): Batch size for feature extraction.

        Returns:
            np.ndarray: Extracted feature embeddings.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = chunked(crops, batch_size)
        data = []
        with torch.no_grad():
            for batch in batches:
                inputs = self.embedding_processor(images=batch, return_tensors="pt").to(DEVICE)
                outputs = self.embedding_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit_clustering_model(self, features: np.ndarray):
        """Fits the clustering model using reduced feature dimensions.

        Args:
            features (np.ndarray): Extracted features.
        """
        projections = self.reducer.fit_transform(features)
        self.clustering_model.fit(projections)

    def predict_teams(self, features: np.ndarray) -> np.ndarray:
        """Predicts team assignments for players based on features.

        Args:
            features (np.ndarray): Extracted features.

        Returns:
            np.ndarray: Predicted team IDs.
        """
        projections = self.reducer.transform(features)
        return self.clustering_model.predict(projections)

    @staticmethod
    def collect_crops(
        player_responses: list[player_inference_pb2.PlayerInferenceResponse], frames
    ):
        """Collects player crops from player inference responses.

        Args:
            player_responses (list[player_inference_pb2.PlayerInferenceResponse]): List of player
                inference responses.
            frames: List of frames.

        Returns:
            list: List of player crops.
        """
        crops = []

        for i, player_response in enumerate(player_responses):
            frame_image = cv2.imdecode(
                np.frombuffer(frames[i].content, np.uint8), cv2.IMREAD_COLOR
            )
            detections: sv.Detections = to_supervision(player_response, frame_image)
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            if detections.class_id is not None:
                detections = detections[detections.class_id == PLAYER_ID]  # type: ignore
                players_crops = [sv.crop_image(frame_image, xyxy) for xyxy in detections.xyxy]
                crops += players_crops

        return crops
