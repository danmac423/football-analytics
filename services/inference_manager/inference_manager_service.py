import copy
import logging
import queue
import threading
from concurrent import futures
from typing import Any, Callable, Generator, Iterator

import cv2
import grpc
import numpy as np
import supervision as sv

from football_analytics.utils.model import to_supervision
from services.ball_inference.grpc_files import ball_inference_pb2, ball_inference_pb2_grpc
from services.config import (
    BALL_INFERENCE_SERVICE_ADDRESS,
    INFERENCE_MANAGER_SERVICE_ADDRESS,
    KEYPOINTS_DETECTION_SERVICE_ADDRESS,
    PLAYER_INFERENCE_SERVICE_ADDRESS,
)
from services.inference_manager.grpc_files import inference_manager_pb2, inference_manager_pb2_grpc
from services.keypoints_detection.grpc_files import (
    keypoints_detection_pb2,
    keypoints_detection_pb2_grpc,
)
from services.player_inference.grpc_files import player_inference_pb2, player_inference_pb2_grpc

BALL_COLOR = "#FF1493"
PLAYER_COLORS = ["#00BFFF", "#FF6347", "#FFD700"]
KEYPOINTS_COLOR = "#FF1493"

DEFAULT_TIMEOUT = 5

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(PLAYER_COLORS), thickness=2)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(PLAYER_COLORS),
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex(BALL_COLOR),
    base=20,
    height=15,
)

VERTEX_ANNOTATOR = sv.VertexAnnotator(color=sv.Color.from_hex(KEYPOINTS_COLOR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/inference_manager_service.log"),
    ],
)
logger = logging.getLogger(__name__)


class InferenceManagerServiceServicer(inference_manager_pb2_grpc.InferenceManagerServiceServicer):
    def __init__(self):
        logger.info("Initializing Inference Manager Service...")
        self.ball_stub = ball_inference_pb2_grpc.YOLOBallInferenceServiceStub(
            grpc.insecure_channel(BALL_INFERENCE_SERVICE_ADDRESS)
        )
        self.player_stub = player_inference_pb2_grpc.YOLOPlayerInferenceServiceStub(
            grpc.insecure_channel(PLAYER_INFERENCE_SERVICE_ADDRESS)
        )
        self.keypoints_stub = keypoints_detection_pb2_grpc.YOLOKeypointsDetectionServiceStub(
            grpc.insecure_channel(KEYPOINTS_DETECTION_SERVICE_ADDRESS)
        )
        logger.info("Inference Manager Service initialized successfully.")

    def _process_responses(
        self, stub: object, method_name: str, queue: queue.Queue, frames: list, thread_name: str
    ) -> None:
        """
        Generic method to process responses from a gRPC service.
        """
        try:
            logger.info(f"Thread {thread_name} started.")
            method = getattr(stub, method_name)
            for response in method(iter(frames)):
                queue.put(response)
            logger.info(f"Thread {thread_name} finished successfully.")
        except Exception as e:
            logger.error(f"Error in {method_name}: {e}")
            queue.put(None)

    def _safe_execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Executes a function and logs errors.
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            return None

    def _get_from_queue(self, queue: queue.Queue, frame_id: int, queue_name: str) -> Any:
        try:
            return queue.get(timeout=DEFAULT_TIMEOUT)
        except queue.Empty:
            logger.warning(
                f"Timeout while waiting for {queue_name} response for frame ID {frame_id}."
            )
            return None

    def _annotate_frame(
        self,
        frame_ndarray: np.ndarray,
        player_response: player_inference_pb2.PlayerInferenceResponse,
        ball_response: ball_inference_pb2.BallInferenceResponse,
        keypoints_response: keypoints_detection_pb2.KeypointsDetectionResponse,
    ) -> np.ndarray:
        """
        Annotates a frame with player, ball, and keypoints data if available.
        """
        annotated_frame = frame_ndarray
        try:
            if player_response is not None:
                detections = to_supervision(player_response, frame_ndarray)
                annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        except Exception as e:
            logger.error(f"Error annotating players: {e}")

        try:
            if ball_response is not None:
                detections = to_supervision(ball_response, frame_ndarray)
                annotated_frame = TRIANGLE_ANNOTATOR.annotate(annotated_frame, detections)
        except Exception as e:
            logger.error(f"Error annotating ball: {e}")

        try:
            if keypoints_response is not None:
                keypoints = to_supervision(keypoints_response, frame_ndarray)
                annotated_frame = VERTEX_ANNOTATOR.annotate(annotated_frame, keypoints)
        except Exception as e:
            logger.error(f"Error annotating keypoints: {e}")

        return annotated_frame

    def ProcessFrames(
        self, request_iterator: Iterator[ball_inference_pb2.Frame], context: grpc.ServicerContext
    ) -> Generator[inference_manager_pb2.Frame, Any, Any]:
        try:
            request_list = list(request_iterator)
        except Exception as e:
            logger.error(f"Faied to read request_iterator: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid input frames.")

        player_queue, ball_queue, keypoints_queue = queue.Queue(), queue.Queue(), queue.Queue()

        threads = {
            "player": threading.Thread(
                target=self._process_responses,
                args=(self.player_stub, "InferencePlayers", player_queue, request_list),
                daemon=True,
            ),
            "ball": threading.Thread(
                target=self._process_responses,
                args=(self.ball_stub, "InferenceBall", ball_queue, request_list),
                daemon=True,
            ),
            "keypoints": threading.Thread(
                target=self._process_responses,
                args=(self.keypoints_stub, "DetectKeypoints", keypoints_queue, request_list),
                daemon=True,
            ),
        }

        for thread_name, thread in threads.items():
            logger.info(f"Starting thread {thread_name}...")
            thread.start()

        for frame_index, frame in enumerate(request_list):
            if not context.is_active():
                logger.warning("Client disconnected. Stopping processing.")
                break

            logger.info(
                f"Processing frame {frame_index + 1}/{len(request_list)} (ID: {frame.frame_id})."
            )

            player_response = self._get_from_queue(player_queue, frame.frame_id, "player")
            ball_response = self._get_from_queue(ball_queue, frame.frame_id, "ball")
            keypoints_response = self._get_from_queue(keypoints_queue, frame.frame_id, "keypoints")

            frame_ndarray = self._safe_execute(
                lambda: cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR)
            )
            if frame_ndarray is None:
                logger.error(f"Failed to decode frame ID {frame.frame_id}. Skipping.")
                continue

            annotated_frame = self._annotate_frame(
                frame_ndarray, player_response, ball_response, keypoints_response
            )

            _, frame_bytes = self._safe_execute(lambda: cv2.imencode(".jpg", annotated_frame))
            if frame_bytes is None:
                logger.error(f"Failed to encode frame ID {frame.frame_id}. Skipping.")
                continue

            yield inference_manager_pb2.Frame(content=frame_bytes.tobytes())

        for thread_name, thread in threads.items():
            thread.join(timeout=10)
            if thread.is_alive():
                logger.error(f"Thread {thread_name} did not terminate properly.")

        logger.info("Finished processing frames.")


def serve():
    """
    serve starts the gRPC server for the Inference Manager service.
    """
    logger.info("Starting Inference Manager Service...")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = InferenceManagerServiceServicer()

    inference_manager_pb2_grpc.add_InferenceManagerServiceServicer_to_server(servicer, server)

    server.add_insecure_port(INFERENCE_MANAGER_SERVICE_ADDRESS)
    logger.info(f"Inference Manager started on {INFERENCE_MANAGER_SERVICE_ADDRESS}.")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
