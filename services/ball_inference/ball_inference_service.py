"""Module that contains the gRPC server for the ball inference service."""

import logging
import os
import signal
from concurrent import futures
from typing import Any, Generator, Iterator

import cv2
import grpc
import numpy as np

from config import BALL_INFERENCE_SERVICE_ADDRESS
from football_analytics.ball_inference.yolo_ball_inferer import YOLOBallInferer
from services.ball_inference.grpc_files import ball_inference_pb2, ball_inference_pb2_grpc

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/ball_inference_service.log")],
)
logger = logging.getLogger(__name__)


class YOLOBallInferenceServiceServicer(ball_inference_pb2_grpc.YOLOBallInferenceServiceServicer):
    """
    Class to implement the gRPC service for the ball inference.

    Attributes:
        inferer (YOLOBallInferer): YOLO Ball Inferer
    """

    def __init__(self):
        self.inferer = YOLOBallInferer()

    def InferenceBall(
        self, request_iterator: Iterator[ball_inference_pb2.Frame], context: grpc.ServicerContext
    ) -> Generator[ball_inference_pb2.BallInferenceResponse, Any, Any]:
        """
        Method that receives a stream of frames and returns a stream of BallInferenceResponse.

        Args:
            request_iterator: The stream of frames.
            context: The context of the gRPC request.

        Yields:
            The BallInferenceResponse for each frame.
        """
        try:
            for frame in request_iterator:
                try:
                    frame_image = cv2.imdecode(
                        np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR
                    )
                    response = self.inferer.infer_ball(frame_image)
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


def shutdown_server(server: grpc.Server, servicer: YOLOBallInferenceServiceServicer):
    """
    Gracefully shuts down the server and logs shutdown events.
    """
    logger.info("Shutting down server...")
    server.stop(grace=1)
    logger.info("Server shut down gracefully.")


def serve():
    """
    Function that starts the gRPC server.
    """
    logger.info("Starting gRPC server...")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = YOLOBallInferenceServiceServicer()

    ball_inference_pb2_grpc.add_YOLOBallInferenceServiceServicer_to_server(servicer, server)

    server.add_insecure_port(BALL_INFERENCE_SERVICE_ADDRESS)

    def handle_signal(signal_num, frame):
        logger.info(f"Received signal {signal_num}. Initiating shutdown...")
        shutdown_server(server, servicer)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    logger.info(f"Server started on {BALL_INFERENCE_SERVICE_ADDRESS}.")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
