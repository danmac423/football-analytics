"""Module for the YOLOBallInferer class."""

import logging

import numpy as np
import supervision as sv
from ultralytics import YOLO
from ultralytics.engine.results import Results

from config import BALL_INFERENCE_MODEL_PATH, DEVICE
from football_analytics.ball_inference.ball_tracker import BallTracker
from services.ball_inference.grpc_files import ball_inference_pb2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/ball_inference_service.log")],
)
logger = logging.getLogger(__name__)


class YOLOBallInferer:
    """
    Class to perform ball inference using YOLO.

    Attributes:
        model (YOLO): The YOLO model.
        tracker (BallTracker): The tracker to track the ball.
    """

    def __init__(self):
        logger.info("Initializing YOLO model...")
        self.model = YOLO(BALL_INFERENCE_MODEL_PATH).to(DEVICE)
        logger.info(f"YOLO model loaded from {BALL_INFERENCE_MODEL_PATH} on device {DEVICE}.")
        self.tracker = self._initialize_tracker()

    def _initialize_tracker(self) -> BallTracker:
        """Initializes the tracker."""
        return BallTracker()

    def reset_tracker(self):
        """Resets the tracker."""
        self.tracker.reset()

    def infer_ball(
        self,
        frame_image: np.ndarray,
    ) -> ball_inference_pb2.BallInferenceResponse:
        """
        Performs ball inference on the given frame image.

        Args:
            frame_image (np.ndarray): The frame image.

        Returns:
            ball_inference_pb2.BallInferenceResponse: The ball inference response
        """
        height, width, _ = frame_image.shape
        slicer = self._create_inference_slicer(width, height)

        detections = self._perform_inference(frame_image, slicer, self.tracker)
        boxes = self._format_detections(detections, width, height)

        return ball_inference_pb2.BallInferenceResponse(boxes=boxes)

    def _create_inference_slicer(self, width: int, height: int) -> sv.InferenceSlicer:
        """
        Creates an inference slicer for the given width and height.

        Args:
            width (int): The width of the frame.
            height (int): The height of the frame.

        Returns:
            sv.InferenceSlicer: The inference slicer.
        """

        def callback(patch: np.ndarray) -> sv.Detections:
            result = self.model(patch, conf=0.3)[0]
            return sv.Detections.from_ultralytics(result)

        slicer = sv.InferenceSlicer(
            callback=callback,
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            slice_wh=(width // 2 + 100, height // 2 + 100),
            overlap_ratio_wh=None,
            overlap_wh=(100, 100),
            iou_threshold=0.1,
        )
        return slicer

    def _perform_inference(
        self, frame_image: np.ndarray, slicer: sv.InferenceSlicer, tracker: BallTracker
    ) -> sv.Detections:
        """
        Performs inference on the given frame image.

        Args:
            frame_image (np.ndarray): The frame image.
            slicer (sv.InferenceSlicer): The inference slicer.
            tracker (BallTracker): The ball tracker.

        Returns:
            sv.Detections: The detections.
        """
        initial_results: Results = self.model(frame_image)[0]
        detections = sv.Detections.from_ultralytics(initial_results)

        detections = slicer(frame_image)
        detections = tracker.update(detections)

        return detections

    def _format_detections(
        self, detections: sv.Detections, width: int, height: int
    ) -> list[ball_inference_pb2.BoundingBox]:
        """
        Fromats the detections to the ball inference response format.

        Args:
            detections (sv.Detections): The detections.
            width (int): The width of the frame.
            height (int): The height of the frame.

        Returns:
            list[ball_inference_pb2.BoundingBox]: The formatted bounding boxes.
        """
        boxes = []
        for box, conf in zip(detections.xyxy, detections.confidence):  # type: ignore
            x1, y1, x2, y2 = box[:4]
            boxes.append(
                ball_inference_pb2.BoundingBox(
                    x1_n=x1 / width,
                    y1_n=y1 / height,
                    x2_n=x2 / width,
                    y2_n=y2 / height,
                    confidence=conf,
                    class_label="ball",
                )
            )
        return boxes
