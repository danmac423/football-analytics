"""Module that contains the gRPC server for the player inference service."""

import logging
import os
import signal
from concurrent import futures
from typing import Any, Generator, Iterator

import cv2
import grpc
import numpy as np

from config import PLAYER_INFERENCE_SERVICE_ADDRESS
from football_analytics.player_inference.yolo_players_inferer import YOLOPlayerInferer
from services.player_inference.grpc_files import player_inference_pb2, player_inference_pb2_grpc

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/player_inference_service.log")],
)
logger = logging.getLogger(__name__)


class YOLOPlayerInferenceServiceServicer(
    player_inference_pb2_grpc.YOLOPlayerInferenceServiceServicer
):
    """
    YOLOPlayerInferenceServiceServicer class to implement the gRPC service.

    Attributes:
        inferer (YOLOPlayerInferer): YOLO Player Inferer
    """

    def __init__(self):
        self.inferer = YOLOPlayerInferer()

    def InferencePlayers(
        self, request_iterator: Iterator[player_inference_pb2.Frame], context: grpc.ServicerContext
    ) -> Generator[player_inference_pb2.PlayerInferenceResponse, Any, Any]:
        """
        Method that receives a stream of frames and returns a stream of PlayerInferenceResponse.

        Args:
            request_iterator (Iterator[player_inference_pb2.Frame]): request iterator
            context (grpc.ServicerContext): context object for the request

        Yields:
            Generator[player_inference_pb2.PlayerInferenceResponse, Any, Any]: returns the response
                with frame_id and boxes
        """
        try:
            for frame in request_iterator:
                try:
                    frame_image = cv2.imdecode(
                        np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR
                    )
                    response = self.inferer.infer_players(frame_image)
                    response.frame_id = frame.frame_id
                    logger.info(
                        f"Frame ID {frame.frame_id} processed with {len(response.boxes)} detections."  # noqa
                    )
                    yield response
                except Exception as e:
                    logger.error(f"Error processing frame ID {frame.frame_id}: {e}")
                    context.abort(grpc.StatusCode.UNKNOWN, str(e))
        finally:
            self.inferer.reset_tracker()


def shutdown_server(server: grpc.Server, servicer: YOLOPlayerInferenceServiceServicer):
    """
    Gracefully shuts down the server and logs shutdown events.
    """
    logger.info("Shutting down server...")
    server.stop(grace=1)
    logger.info("Server shut down gracefully.")


def serve():
    """
    Function to start the gRPC server for the YOLO Player Detection Service.
    """
    logger.info("Starting gRPC server...")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = YOLOPlayerInferenceServiceServicer()

    player_inference_pb2_grpc.add_YOLOPlayerInferenceServiceServicer_to_server(servicer, server)

    server.add_insecure_port(PLAYER_INFERENCE_SERVICE_ADDRESS)

    def handle_signal(signal_num, frame):
        logger.info(f"Received signal {signal_num}. Initiating shutdown...")
        shutdown_server(server, servicer)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Server started on {PLAYER_INFERENCE_SERVICE_ADDRESS}.")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
