"""Module that contains the gRPC server for the keypoints detection service."""

import logging
import os
import signal
from concurrent import futures
from typing import Any, Generator, Iterator

import grpc

from config import KEYPOINTS_DETECTION_SERVICE_ADDRESS
from football_analytics.keypoint_detection.yolo_keypoints_detector import YOLOKeypointsDetector
from services.keypoints_detection.grpc_files import (
    keypoints_detection_pb2,
    keypoints_detection_pb2_grpc,
)

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/keypoints_detection_service.log"),
    ],
)
logger = logging.getLogger(__name__)


class YOLOKeypointsDetectionServiceServicer(
    keypoints_detection_pb2_grpc.YOLOKeypointsDetectionServiceServicer
):
    """
    YOLOKeypointsDetectionServiceServicer class to implement the gRPC service.

    Attributes:
        detector (YOLOKeypointsDetector): YOLO Keypoints Detector
    """

    def __init__(self):
        self.detector = YOLOKeypointsDetector()

    def DetectKeypoints(
        self,
        request_iterator: Iterator[keypoints_detection_pb2.Frame],
        context: grpc.ServicerContext,
    ) -> Generator[keypoints_detection_pb2.KeypointsDetectionResponse, Any, Any]:
        """
        Method that receives a stream of frames and returns a stream of KeypointsDetectionResponse.

        Args:
            request_iterator (Iterator[keypoints_detection_pb2.Frame]): request iterator
            context (grpc.ServicerContext): context object for the request

        Yields:
            Generator[keypoints_detection_pb2.KeypointsDetectionResponse, Any, Any]: returns the
                response with frame_id and keypoints
        """
        for frame in request_iterator:
            try:
                keypoints_detection_response = self.detector.detect_keypoints(frame)

                keypoints_detection_response.frame_id = frame.frame_id
                logger.info(
                    f"Frame ID {frame.frame_id} processed with {len(keypoints_detection_response.boxes)} detections."  # noqa
                )

                yield keypoints_detection_response

            except Exception as e:
                logger.error(f"Error processing frame ID {frame.frame_id}: {e}")
                context.abort(grpc.StatusCode.UNKNOWN, str(e))


def shutdown_server(server: grpc.Server, servicer: YOLOKeypointsDetectionServiceServicer):
    """
    Gracefully shuts down the server and logs shutdown events.
    """
    logger.info("Shutting down server...")
    server.stop(grace=1)
    logger.info("Server shut down gracefully.")


def serve():
    """
    Function that starts the gRPC server and adds the YOLOKeypointsDetectionServiceServicer
    """
    logger.info("Starting gRPC server...")

    servicer = YOLOKeypointsDetectionServiceServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))

    keypoints_detection_pb2_grpc.add_YOLOKeypointsDetectionServiceServicer_to_server(
        servicer, server
    )

    server.add_insecure_port(KEYPOINTS_DETECTION_SERVICE_ADDRESS)

    def handle_signal(signal_num, frame):
        logger.info(f"Received signal {signal_num}. Initiating shutdown...")
        shutdown_server(server, servicer)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Server started on {KEYPOINTS_DETECTION_SERVICE_ADDRESS}.")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
