"""Module responsible for tracking players using YOLO."""

import os
from typing import List

import numpy as np
import supervision as sv
from ultralytics import YOLO
from ultralytics.engine.results import Results

from config import DEVICE, PLAYER_INFERENCE_MODEL_PATH
from services.player_inference.grpc_files import player_inference_pb2

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"


class YOLOPlayerInferer:
    """
    Class to perform player inference using YOLO.

    Attributes:
        model (YOLO): The YOLO model.
        tracker (sv.ByteTrack): The tracker to track players.
    """

    def __init__(self):
        self.model = self._load_model()
        self.tracker = self._initialize_tracker()

    def _load_model(self) -> YOLO:
        """Loads the YOLO model.

        Returns:
            YOLO: The loaded YOLO model.
        """
        model = YOLO(PLAYER_INFERENCE_MODEL_PATH).to(DEVICE)
        return model

    def _initialize_tracker(self) -> sv.ByteTrack:
        """Initializes the tracker.

        Returns:
            sv.ByteTrack: The initialized tracker.
        """
        return sv.ByteTrack(
            lost_track_buffer=100,
            minimum_consecutive_frames=3,
        )

    def _normalize_box(self, box: List[float], width: int, height: int) -> List[float]:
        """Normalizes the bounding box coordinates to a scale of 0 to 1.

        Args:
            box (List[float]): The bounding box coordinates [x1, y1, x2, y2].
            width (int): Width of the frame.
            height (int): Height of the frame.

        Returns:
            List[float]: Normalized bounding box coordinates [x1_n, y1_n, x2_n, y2_n].
        """
        x1, y1, x2, y2 = box[:4]
        return [x1 / width, y1 / height, x2 / width, y2 / height]

    def reset_tracker(self):
        self.tracker.reset()

    def infer_players(
        self, frame_image: np.ndarray
    ) -> player_inference_pb2.PlayerInferenceResponse:
        """Processes a single frame and returns inference results.

        Args:
            frame_image (np.ndarray): The input frame image.

        Returns:
            player_inference_pb2.PlayerInferenceResponse: The response containing detected players.
        """
        height, width, _ = frame_image.shape
        results: Results = self.model(frame_image)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        labels = results.names
        boxes = []

        for box, conf, cls, tracker_id in zip(
            detections.xyxy,
            detections.confidence,  # type: ignore
            detections.class_id,  # type: ignore
            detections.tracker_id,  # type: ignore
        ):
            x1_n, y1_n, x2_n, y2_n = self._normalize_box(box, width, height)

            boxes.append(
                player_inference_pb2.BoundingBox(
                    x1_n=x1_n,
                    y1_n=y1_n,
                    x2_n=x2_n,
                    y2_n=y2_n,
                    confidence=conf,
                    class_label=labels[cls],
                    tracker_id=tracker_id,
                )
            )

        return player_inference_pb2.PlayerInferenceResponse(boxes=boxes)
