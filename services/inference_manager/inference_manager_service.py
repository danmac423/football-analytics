import logging
import os
import queue
import signal
import threading
from concurrent import futures
from queue import Empty
from time import time
from typing import Any, Callable, Generator, Iterator

import cv2
import grpc
import numpy as np
import supervision as sv

from football_analytics.annotations.radar import generate_radar
from football_analytics.football_pitch.football_pitch_configuration import (
    FootballPitchConfiguration,
)
from football_analytics.football_pitch.view_transformer import ViewTransformer
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

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

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

        self.ball_channel = grpc.insecure_channel(BALL_INFERENCE_SERVICE_ADDRESS)
        self.player_channel = grpc.insecure_channel(PLAYER_INFERENCE_SERVICE_ADDRESS)
        self.keypoints_channel = grpc.insecure_channel(KEYPOINTS_DETECTION_SERVICE_ADDRESS)

        self.ball_stub = ball_inference_pb2_grpc.YOLOBallInferenceServiceStub(self.ball_channel)
        self.player_stub = player_inference_pb2_grpc.YOLOPlayerInferenceServiceStub(
            self.player_channel
        )
        self.keypoints_stub = keypoints_detection_pb2_grpc.YOLOKeypointsDetectionServiceStub(
            self.keypoints_channel
        )

        self.threads = {}
        self.stop_event = threading.Event()

        self.previous_positions = {}
        self.view_transformer = None

        logger.info("Inference Manager Service initialized successfully.")

    def close_connections(self):
        """
        Closes all gRPC connections to external services.
        """
        logger.info("Closing gRPC connections to external services...")
        self.ball_channel.close()
        self.player_channel.close()
        self.keypoints_channel.close()
        logger.info("All gRPC connections closed.")

    def _process_responses(
        self,
        stub: object,
        method_name: str,
        queue: queue.Queue,
        frames: list,
        thread_name: str,
        context: grpc.ServicerContext,
    ) -> None:
        """
        Generic method to process responses from a gRPC service.
        """
        try:
            logger.info(f"Thread {thread_name} started.")
            method = getattr(stub, method_name)
            for response in method(iter(frames)):
                if self.stop_event.is_set() or not context.is_active():
                    logger.info(f"Thread {thread_name} stopping due to shutdown signal.")
                    break
                queue.put(response)
            logger.info(f"Thread {thread_name} finished successfully.")
        except grpc.RpcError as e:
            logger.warning(f"Error in {method_name}: {e.details()}")
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
            while not self.stop_event.is_set():
                return queue.get(timeout=DEFAULT_TIMEOUT)
        except Empty:
            if self.stop_event.is_set():
                logger.info(f"Server shutdown detected. Stopping wait for {queue_name}.")
                return None
            logger.warning(
                f"Timeout while waiting for {queue_name} response for frame ID {frame_id}."
            )
            return None

    def _initialize_view_transformer(
            self,
            frame: np.ndarray,
            keypoints_response: keypoints_detection_pb2.KeypointsDetectionResponse\
        ) -> None:
        """
        Initializes the ViewTransformer with the reference points from the first frame.

        Args:
            frame (np.ndarray): The first frame.
            keypoints_response (keypoints_detection_pb2.KeypointsDetectionResponse): The keypoints
                response from YOLO service.

        Raises:
            ValueError: If keypoints are not available.
        """
        if not keypoints_response or not keypoints_response.keypoints:
            raise ValueError("Keypoints not available for initializing ViewTransformer.")

        keypoints = to_supervision(keypoints_response, frame)
        filter = keypoints.confidence[0] > 0.5
        frame_reference_points = keypoints.xy[0][filter]

        config = FootballPitchConfiguration()
        pitch_reference_points = np.array(config.vertices, dtype=np.float32)[filter]

        self.view_transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points,
        )


    def _calculate_velocity(
            self,
            player_id: int,
            current_position: tuple[float, float]
        ) -> float:

        real_position = self.view_transformer.transform_points(np.array([current_position]))[0]

        if player_id in self.previous_positions:
            prev_position = self.previous_positions[player_id]

            prev_real_position = self.view_transformer.transform_points(
                np.array([prev_position])
            )[0]

            distance = np.sqrt((real_position[0] - prev_real_position[0]) ** 2 +
                           (real_position[1] - prev_real_position[1]) ** 2)

            delta_time = 1/25
            velocity = (distance / 100.0) / delta_time if delta_time > 0 else 0
        else:
            velocity = 0

        self.previous_positions[player_id] = current_position
        return velocity

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
        annotated_frame = frame_ndarray.copy()
        current_time = time()

        try:
            if player_response is not None:
                detections = to_supervision(player_response, frame_ndarray)

                for i, detection in enumerate(detections.xyxy):
                    tracker_id = int(detections.tracker_id[i])
                    position = (
                        (detection[0] + detection[2]) / 2,  # x1 + x2 / 2
                        (detection[1] + detection[3]) / 2,  # y1 + y2 / 2
                    )
                    velocity = self._calculate_velocity(tracker_id, position)
                    annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)

                    velocity_text = f"{velocity:.2f} m/s"
                    position = (int(detection[0]), int(detection[1]))
                    cv2.putText(
                        annotated_frame,
                        velocity_text,
                        position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
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

    def _generate_radar(
            self,
            frame: np.ndarray,
            player_response: player_inference_pb2.PlayerInferenceResponse,
            ball_response: ball_inference_pb2.BallInferenceResponse,
            keypoints_response: keypoints_detection_pb2.KeypointsDetectionResponse
        ) -> np.ndarray:

        if not keypoints_response or not keypoints_response.keypoints:
            return frame

        if not player_response:
            player_detections = sv.Detections(
                xyxy = np.empty((0, 4)),
            )
        else:
            player_detections = to_supervision(player_response, frame)

        if not ball_response:
            ball_detections = sv.Detections(
                xyxy = np.empty((0, 4))
            )
        else:
            ball_detections = to_supervision(ball_response, frame)

        return generate_radar(
            frame,
            player_detections,
            ball_detections,
            to_supervision(keypoints_response, frame)
        )


    def ProcessFrames(
        self, request_iterator: Iterator[ball_inference_pb2.Frame], context: grpc.ServicerContext
    ) -> Generator[inference_manager_pb2.Frame, Any, Any]:
        client_address = context.peer()
        logger.info(f"Client connected: {client_address}")

        try:
            request_list = list(request_iterator)
        except Exception as e:
            logger.error(f"Faied to read request_iterator: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid input frames.")

        player_queue, ball_queue, keypoints_queue = queue.Queue(), queue.Queue(), queue.Queue()

        self.threads = {
            "player": threading.Thread(
                target=self._process_responses,
                args=(
                    self.player_stub,
                    "InferencePlayers",
                    player_queue,
                    request_list,
                    "player",
                    context,
                ),
                daemon=True,
            ),
            "ball": threading.Thread(
                target=self._process_responses,
                args=(self.ball_stub, "InferenceBall", ball_queue, request_list, "ball", context),
                daemon=True,
            ),
            "keypoints": threading.Thread(
                target=self._process_responses,
                args=(
                    self.keypoints_stub,
                    "DetectKeypoints",
                    keypoints_queue,
                    request_list,
                    "keypoints",
                    context,
                ),
                daemon=True,
            ),
        }

        for thread_name, thread in self.threads.items():
            logger.info(f"Starting thread {thread_name}...")
            thread.start()

        for frame_index, frame in enumerate(request_list):
            if not context.is_active() or self.stop_event.is_set():
                logger.info("Server shutdown detected. Closing client connection.")
                context.abort(grpc.StatusCode.CANCELLED, "Server is shutting down.")
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

            if self.view_transformer is None:
                self._initialize_view_transformer(frame_ndarray, keypoints_response)

            annotated_frame = self._annotate_frame(
                frame_ndarray, player_response, ball_response, keypoints_response
            )

            annotated_frame = self._generate_radar(
                annotated_frame,
                player_response,
                ball_response,
                keypoints_response
            )

            _, frame_bytes = self._safe_execute(lambda: cv2.imencode(".jpg", annotated_frame))
            if frame_bytes is None:
                logger.error(f"Failed to encode frame ID {frame.frame_id}. Skipping.")
                continue

            yield inference_manager_pb2.Frame(
                content=frame_bytes.tobytes(), frame_id=frame.frame_id
            )

        for thread_name, thread in self.threads.items():
            thread.join(timeout=10)
            if thread.is_alive():
                logger.error(f"Thread {thread_name} did not terminate properly.")

        logger.info(f"Finished processing frames for client: {client_address}")


def shutdown_server(server, servicer: InferenceManagerServiceServicer):
    """
    Gracefully shuts down the server and waits for threads to complete.
    """
    logger.info("Shutting down server...")

    servicer.stop_event.set()
    servicer.close_connections()

    for thread_name, thread in servicer.threads.items():
        logger.info(f"Waiting for thread {thread_name} to finish...")
        thread.join()
        if thread.is_alive():
            logger.error(f"Thread {thread_name} did not terminate properly.")

    server.stop(grace=5)

    logger.info("All threads finished. Server shut down gracefully.")


def serve():
    """
    serve starts the gRPC server for the Inference Manager service.
    """
    logger.info("Starting Inference Manager Service...")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = InferenceManagerServiceServicer()

    inference_manager_pb2_grpc.add_InferenceManagerServiceServicer_to_server(servicer, server)

    server.add_insecure_port(INFERENCE_MANAGER_SERVICE_ADDRESS)

    def handle_signal(signal_num, frame):
        logger.info(f"Received signal {signal_num}. Initiating shutdown...")
        shutdown_server(server, servicer)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Inference Manager started on {INFERENCE_MANAGER_SERVICE_ADDRESS}.")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
