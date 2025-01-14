import logging
import os
from concurrent import futures
from typing import Any, Generator, Iterator

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

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

BALL_COLOR = "#FF1493"
PLAYER_COLORS = ["#00BFFF", "#FF6347", "#FFD700"]
KEYPOINTS_COLOR = "#FF1493"

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
    """
    InferenceManagerServiceServicer is the class that implements the gRPC server
    for the Inference Manager service.

    Attributes:
        ball_stub: The gRPC stub for the Ball Inference service.
        player_stub: The gRPC stub for the Player Inference service.
        keypoints_stub: The gRPC stub for the Keypoints Detection service.
    """

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

    def ProcessFrames(
        self, request_iterator: Iterator[ball_inference_pb2.Frame], context: grpc.ServicerContext
    ) -> Generator[inference_manager_pb2.Frame, Any, Any]:
        """
        ProcessFrames is the gRPC method that processes the frames by calling the Ball Inference,
        Player Inference, and Keypoints Detection services.

        Args:
            request_iterator (Iterator[ball_inference_pb2.Frame]): The iterator of frames
            to process.
            context (grpc.ServicerContext): The context of the gRPC request.
        """
        for frame in request_iterator:
            logger.debug(f"Processing frame ID: {frame.frame_id}")

            ball_response: ball_inference_pb2.BallInferenceResponse = next(
                self.ball_stub.InferenceBall(iter([frame]))
            )
            player_response: player_inference_pb2.PlayerInferenceResponse = next(
                self.player_stub.InferencePlayers(iter([frame]))
            )
            keypoints_response: keypoints_detection_pb2.KeypointsDetectionResponse = next(
                self.keypoints_stub.DetectKeypoints(iter([frame]))
            )

            frame_ndarray = cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR)

            try:
                detections = to_supervision(player_response, frame_ndarray)

                annotated_frame = ELLIPSE_ANNOTATOR.annotate(frame_ndarray, detections)
            except Exception as e:
                logger.error(f"Error annotating players for frame ID {frame.frame_id}: {e}")
                continue


            try:
               detections = to_supervision(ball_response, frame_ndarray)

               annotated_frame = TRIANGLE_ANNOTATOR.annotate(annotated_frame, detections)
            except Exception as e:
                logger.error(f"Error annotating ball for frame ID {frame.frame_id}: {e}")
                continue


            try:
                keypoints = to_supervision(keypoints_response, frame_ndarray)

                annotated_frame = VERTEX_ANNOTATOR.annotate(annotated_frame, keypoints)
            except Exception as e:
                logger.error(f"Error annotating keypoints for frame ID {frame.frame_id}: {e}")
                continue

            try:
                _, frame_bytes = cv2.imencode(".jpg", annotated_frame)
                logger.debug(f"Frame ID {frame.frame_id} encoded successfully.")
                yield inference_manager_pb2.Frame(content=frame_bytes.tobytes())
            except Exception as e:
                logger.error(f"Error encoding frame ID {frame.frame_id}: {e}")
                continue

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
