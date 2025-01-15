import logging
import os
import signal
from concurrent import futures
from typing import Any, Generator, Iterator

import cv2
import grpc
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from services.ball_inference.grpc_files import ball_inference_pb2, ball_inference_pb2_grpc
from services.config import BALL_INFERENCE_MODEL_PATH, BALL_INFERENCE_SERVICE_ADDRESS, DEVICE

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/ball_inference_service.log")],
)
logger = logging.getLogger(__name__)


class YOLOBallInferenceServiceServicer(ball_inference_pb2_grpc.YOLOBallInferenceServiceServicer):
    """
    Class that implements the YOLOBallInferenceServiceServicer from the gRPC generated files.

    Attributes:
        model: The YOLO model to be used for inference.
    """

    def __init__(self):
        logger.info("Initializing YOLO model...")
        self.model = YOLO(BALL_INFERENCE_MODEL_PATH).to(DEVICE)
        logger.info(f"YOLO model loaded from {BALL_INFERENCE_MODEL_PATH} on device {DEVICE}.")

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
        for frame in request_iterator:
            try:
                frame_image = cv2.imdecode(
                    np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR
                )

                result: Results = self.model.predict(frame_image)[0]
                labels = result.names

                boxes = []

                for box in result.boxes:
                    xyxyn = box.xyxyn.cpu().flatten()

                    boxes.append(
                        ball_inference_pb2.BoundingBox(
                            x1_n=xyxyn[0].item(),
                            y1_n=xyxyn[1].item(),
                            x2_n=xyxyn[2].item(),
                            y2_n=xyxyn[3].item(),
                            confidence=box.conf.item(),
                            class_label=labels[int(box.cls.item())],
                        )
                    )

                logger.info(f"Frame ID {frame.frame_id} processed with {len(boxes)} detections.")
                yield ball_inference_pb2.BallInferenceResponse(
                    frame_id=frame.frame_id, boxes=boxes
                )
            except Exception as e:
                logger.error(f"Error processing frame ID {frame.frame_id}: {e}")
                context.abort(grpc.StatusCode.UNKNOWN, str(e))


def shutdown_server(server, servicer):
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
