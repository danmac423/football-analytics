"""
Module for the Inference Manager service. This service is responsible for managing the inference
services for players, ball, and keypoints detection.
"""

import logging
import os
import queue
import signal
import threading
from concurrent import futures
from queue import Empty
from typing import Any, Callable, Generator, Iterator

import cv2
import grpc
import numpy as np

from config import (
    BALL_INFERENCE_SERVICE_ADDRESS,
    INFERENCE_MANAGER_SERVICE_ADDRESS,
    KEYPOINTS_DETECTION_SERVICE_ADDRESS,
    PLAYER_INFERENCE_SERVICE_ADDRESS,
)
from football_analytics.annotations.frame_annotator import FrameAnnotator
from football_analytics.camera_estimation.velocity import VelocityEstimator
from football_analytics.camera_estimation.view_transformer import ViewTransformer
from services.ball_inference.grpc_files import ball_inference_pb2, ball_inference_pb2_grpc
from services.inference_manager.grpc_files import inference_manager_pb2, inference_manager_pb2_grpc
from services.keypoints_detection.grpc_files import (
    keypoints_detection_pb2_grpc,
)
from services.player_inference.grpc_files import player_inference_pb2_grpc

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"


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
    """
    Class to implement the gRPC service for the Inference Manager. This service is responsible for
    managing the inference services for players, ball, and keypoints detection.

    Attributes:
        ball_channel (grpc.Channel): gRPC channel for the ball inference service
        player_channel (grpc.Channel): gRPC channel for the player inference service
        keypoints_channel (grpc.Channel): gRPC channel for the keypoints detection service
        ball_stub (ball_inference_pb2_grpc.YOLOBallInferenceServiceStub): gRPC stub for the ball
            inference service
        player_stub (player_inference_pb2_grpc.YOLOPlayerInferenceServiceStub): gRPC stub for the
            player inference service
        keypoints_stub (keypoints_detection_pb2_grpc.YOLOKeypointsDetectionServiceStub): gRPC stub
            for the keypoints detection service
        queues (dict[str, queue.Queue]): dictionary of queues for the responses from the inference
            services
        threads (dict): dictionary of threads for processing responses from the inference services
        stop_event (threading.Event): event to signal the threads to stop
        frame_annotator (FrameAnnotator): FrameAnnotator object for annotating frames
        velocity_estimator (VelocityEstimator): VelocityEstimator object for estimating velocities
    """

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

        self.queues: dict[str, queue.Queue] = {
            "ball": queue.Queue(),
            "players": queue.Queue(),
            "keypoints": queue.Queue(),
        }

        self.threads = {}
        self.stop_event = threading.Event()

        self.frame_annotator = FrameAnnotator()

        self.velocity_estimator = VelocityEstimator()

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
    ):
        """
        Generic method to process responses from a gRPC service.

        Args:
            stub (object): The gRPC stub to call the method on.
            method_name (str): The name of the method to call.
            queue (queue.Queue): The queue to put the responses in.
            frames (list): The list of frames to process.
            thread_name (str): The name of the thread.
            context (grpc.ServicerContext): The context of the gRPC request.
        """
        try:
            logger.info(f"Thread {thread_name} started.")
            method = getattr(stub, method_name)
            for response in method(iter(frames)):
                if self.stop_event.is_set() or not context.is_active():
                    self.queues[thread_name].put(None)
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
        """
        Gets an item from a queue and logs errors.

        Args:
            queue (queue.Queue): The queue to get the item from.
            frame_id (int): The frame ID to get the item for.
            queue_name (str): The name of the queue.

        Returns:
            Any: The item from the queue
        """
        try:
            while not self.stop_event.is_set():
                item = queue.get()
                if item is None:
                    logger.info(f"Received termination signal for {queue_name}.")
                    return None
                return item
        except Empty:
            if self.stop_event.is_set():
                logger.info(f"Server shutdown detected. Stopping wait for {queue_name}.")
                return None
            logger.warning(
                f"Timeout while waiting for {queue_name} response for frame ID {frame_id}."
            )
            return None

    def ProcessFrames(
        self, request_iterator: Iterator[ball_inference_pb2.Frame], context: grpc.ServicerContext
    ) -> Generator[inference_manager_pb2.Frame, Any, Any]:
        """
        Main method to process frames and return annotated frames. This method is called by the
        client to process frames and return annotated frames.

        Args:
            request_iterator (Iterator[ball_inference_pb2.Frame]): The iterator for the request.
            context (grpc.ServicerContext): The context of the gRPC request.

        Yields:
            Generator[inference_manager_pb2.Frame, Any, Any]: The annotated frames
        """

        client_address = context.peer()
        logger.info(f"Client connected: {client_address}")

        try:
            request_list = list(request_iterator)
        except Exception as e:
            logger.error(f"Faied to read request_iterator: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid input frames.")

        self.threads = {
            "player": threading.Thread(
                target=self._process_responses,
                args=(
                    self.player_stub,
                    "InferencePlayers",
                    self.queues["players"],
                    request_list,
                    "players",
                    context,
                ),
                daemon=True,
            ),
            "ball": threading.Thread(
                target=self._process_responses,
                args=(
                    self.ball_stub,
                    "InferenceBall",
                    self.queues["ball"],
                    request_list,
                    "ball",
                    context,
                ),
                daemon=True,
            ),
            "keypoints": threading.Thread(
                target=self._process_responses,
                args=(
                    self.keypoints_stub,
                    "DetectKeypoints",
                    self.queues["keypoints"],
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

            player_response = self._get_from_queue(
                self.queues["players"], frame.frame_id, "players"
            )
            ball_response = self._get_from_queue(self.queues["ball"], frame.frame_id, "ball")
            keypoints_response = self._get_from_queue(
                self.queues["keypoints"], frame.frame_id, "keypoints"
            )

            frame_ndarray = self._safe_execute(
                lambda: cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR)
            )
            if frame_ndarray is None:
                logger.error(f"Failed to decode frame ID {frame.frame_id}. Skipping.")
                continue

            annotated_frame = frame_ndarray.copy()

            fps = frame.fps
            delta_time = 1 / fps

            if keypoints_response:
                view_transformer = ViewTransformer.get_view_transformer(
                    frame_ndarray, keypoints_response
                )
                velocities = self.velocity_estimator.estimate_velocities(
                    view_transformer, player_response, frame_ndarray, delta_time
                )
            else:
                velocities = {}
            try:
                annotated_frame = self.frame_annotator.annotate_frame(
                    annotated_frame,
                    player_response,
                    ball_response,
                    keypoints_response,
                    velocities,
                )
                annotated_frame = self.frame_annotator.generate_radar(
                    annotated_frame, player_response, ball_response, keypoints_response
                )
            except Exception as e:
                logger.error(f"Failed to annotate frame ID {frame.frame_id}: {e}")

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


def shutdown_server(server: grpc.Server, servicer: InferenceManagerServiceServicer):
    """
    Gracefully shuts down the server and waits for threads to complete.

    Args:
        server (grpc.Server): The gRPC server to shut down.
        servicer (InferenceManagerServiceServicer): The servicer object to stop.
    """
    logger.info("Shutting down server...")

    servicer.stop_event.set()
    servicer.close_connections()

    for thread_name, thread in servicer.threads.items():
        logger.info(f"Waiting for thread {thread_name} to finish...")
        thread.join()
        if thread.is_alive():
            logger.error(f"Thread {thread_name} did not terminate properly.")

    for q in servicer.queues.values():
        q.put(None)

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
