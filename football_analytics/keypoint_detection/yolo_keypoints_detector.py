"""Module for the YOLOKeypointsDetector class."""

import logging
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from config import DEVICE, KEYPOINTS_DETECTION_MODEL_PATH
from services.keypoints_detection.grpc_files import keypoints_detection_pb2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/keypoints_detection_service.log"),
    ],
)
logger = logging.getLogger(__name__)


class YOLOKeypointsDetector:
    """
    YOLOKeypointsDetector is a class that wrappes the YOLO model for keypoints detection.

    Attributes:
        model (YOLO): The YOLO model for keypoints detection.
    """

    def __init__(self):
        logger.info("Initializing YOLO model...")
        self.model = YOLO(KEYPOINTS_DETECTION_MODEL_PATH).to(DEVICE)
        logger.info(f"YOLO model loaded from {KEYPOINTS_DETECTION_MODEL_PATH} on device {DEVICE}.")

    def detect_keypoints(
        self, frame: keypoints_detection_pb2.Frame
    ) -> keypoints_detection_pb2.KeypointsDetectionResponse:
        """
        Detects keypoints in the given frame.

        Args:
            frame (keypoints_detection_pb2.Frame): The frame to detect keypoints in.

        Returns:
            keypoints_detection_pb2.KeypointsDetectionResponse: Response with the detected
                keypoints.
        """
        frame_image = self._decode_frame(frame.content)

        frame_image_resized = cv2.resize(frame_image, (640, 640))

        results: Results = self.model(frame_image_resized)[0]

        boxes = self._extract_boxes(results)
        keypoints = self._extract_keypoints(results, frame_image.shape, frame_image_resized.shape)  # type: ignore

        logger.info(
            f"Frame ID {frame.frame_id} processed with {len(keypoints)} keypoints and "
            f"{len(boxes)} detections."
        )

        return keypoints_detection_pb2.KeypointsDetectionResponse(
            frame_id=frame.frame_id, boxes=boxes, keypoints=keypoints
        )

    def _decode_frame(self, content: bytes) -> np.ndarray:
        """Decodes a video frame from bytes.

        Args:
            content (bytes): The frame content.

        Returns:
            np.ndarray: The decoded frame.
        """
        return cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

    def _extract_boxes(self, results: Results) -> List[keypoints_detection_pb2.BoundingBox]:
        """Extracts bounding boxes from YOLO results.

        Args:
            results (Results): The YOLO results.

        Returns:
            List[keypoints_detection_pb2.BoundingBox]: The extracted bounding boxes.
        """
        boxes = []
        labels = results.names

        for box in results.boxes:
            coordinates = box.xyxyn.cpu().numpy().flatten()
            x1_n, y1_n, x2_n, y2_n = coordinates[:4]
            boxes.append(
                keypoints_detection_pb2.BoundingBox(
                    x1_n=x1_n,
                    y1_n=y1_n,
                    x2_n=x2_n,
                    y2_n=y2_n,
                    confidence=box.conf.item(),
                    class_label=labels[int(box.cls.item())],
                )
            )

        return boxes

    def _extract_keypoints(
        self,
        results: Results,
        original_shape: Tuple[int, int, int],
        resized_shape: Tuple[int, int, int],
    ) -> List[keypoints_detection_pb2.Keypoint]:
        """Extracts keypoints from YOLO results.

        Args:
            results (Results): The YOLO results.
            original_shape (Tuple[int, int, int]): The original frame shape.
            resized_shape (Tuple[int, int, int]): The resized frame shape.

        Returns:
            List[keypoints_detection_pb2.Keypoint]: The extracted keypoints.
        """
        keypoints = []
        original_height, original_width, _ = original_shape
        height, width, _ = resized_shape

        for keypoint in results.keypoints:
            for point in keypoint.data.cpu().numpy()[0]:
                x, y = point[:2]
                x = x / width * original_width
                y = y / height * original_height
                confidence = point[2] if len(point) > 2 else 0.0
                keypoints.append(
                    keypoints_detection_pb2.Keypoint(
                        x=float(x), y=float(y), confidence=float(confidence)
                    )
                )

        return keypoints
