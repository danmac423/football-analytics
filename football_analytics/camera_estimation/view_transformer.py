"""Module for transforming points from camera view to pitch view."""

import cv2
import numpy as np

from football_analytics.football_pitch.football_pitch_configuration import (
    FootballPitchConfiguration,
)
from football_analytics.utils.model import to_supervision
from services.keypoints_detection.grpc_files import keypoints_detection_pb2


class ViewTransformer:
    """
    ViewTransformer is a class that transforms points from camera view to pitch view.

    Attributes:
        metrics (np.ndarray): The transformation matrix.

    Args:
        source (np.ndarray): The source points. For example, the points in the camera view.
        target (np.ndarray): The target points. For example, the points in the pitch view.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)

        self.metrics, _ = cv2.findHomography(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms the given points from camera view to pitch view.

        Args:
            points (np.ndarray): The points to transform.

        Returns:
            np.ndarray: The transformed points.
        """
        if points.size == 0:
            return np.array([])
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.metrics)

        return points.reshape(-1, 2).astype(np.float32)

    @classmethod
    def get_view_transformer(
        cls,
        frame: np.ndarray,
        keypoints_response: keypoints_detection_pb2.KeypointsDetectionResponse,
    ) -> "ViewTransformer":
        """
        Initializes the ViewTransformer using the keypoints from the keypoints detection response.

        Args:
            frame (np.ndarray): The frame in ndarray format.
            keypoints_response (keypoints_detection_pb2.KeypointsDetectionResponse): The keypoints
                detection response.

        Returns:
            ViewTransformer: The initialized ViewTransformer.
        """
        if not keypoints_response or not keypoints_response.keypoints:
            raise ValueError("Keypoints not available for initializing ViewTransformer.")

        keypoints = to_supervision(keypoints_response, frame)
        filter = keypoints.confidence[0] > 0.5
        frame_reference_points = keypoints.xy[0][filter]

        config = FootballPitchConfiguration()
        pitch_reference_points = np.array(config.vertices, dtype=np.float32)[filter]

        return cls(
            source=frame_reference_points,
            target=pitch_reference_points,
        )
