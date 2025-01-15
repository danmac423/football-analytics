import logging
import os
import signal
from concurrent import futures
from typing import Any, Generator, Iterator

import cv2
import grpc
import numpy as np
import supervision as sv
from ultralytics import YOLO
from ultralytics.engine.results import Results

from services.config import DEVICE, PLAYER_INFERENCE_MODEL_PATH, PLAYER_INFERENCE_SERVICE_ADDRESS
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
    """YOLOPlayerInferenceServiceServicer class to implement the gRPC service.

    Attributes:
        model (YOLO): YOLO model object
    """

    def __init__(self):
        logger.info("Initializing YOLO model...")
        self.model = YOLO(PLAYER_INFERENCE_MODEL_PATH).to(DEVICE)
        logger.info(f"YOLO model loaded from {PLAYER_INFERENCE_MODEL_PATH} on device {DEVICE}.")

    def InferencePlayers(
        self, request_iterator: Iterator[player_inference_pb2.Frame], context: grpc.ServicerContext
    ) -> Generator[player_inference_pb2.PlayerInferenceResponse, Any, Any]:
        """InferencePlayers method for the gRPC service which takes a stream of frames
        and returns the response with the bounding boxes.

        Args:
            request_iterator (Iterator[player_inference_pb2.Frame]): request iterator
            context (grpc.ServicerContext): context object for the request

        Yields:
            Generator[player_inference_pb2.PlayerInferenceResponse, Any, Any]: returns the response
            with frame_id and boxes
        """
        tracker = sv.ByteTrack(
            lost_track_buffer=100,
            minimum_consecutive_frames=3,
        )

        for frame in request_iterator:
            try:
                frame_image = cv2.imdecode(
                    np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR
                )
                height, width, _ = frame_image.shape

                results: Results = self.model(frame_image)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = tracker.update_with_detections(detections)

                labels = results.names

                boxes = []

                for box, conf, cls, tracker_id in zip(
                    detections.xyxy,
                    detections.confidence,
                    detections.class_id,
                    detections.tracker_id,
                ):
                    x1, y1, x2, y2 = box[:4]
                    x1_n = x1 / width
                    y1_n = y1 / height
                    x2_n = x2 / width
                    y2_n = y2 / height

                    boxes.append(
                        player_inference_pb2.BoundingBox(
                            x1_n=x1_n,
                            y1_n=y1_n,
                            x2_n=x2_n,
                            y2_n=y2_n,
                            confidence=conf,
                            class_label=labels[cls],
                            tracker_id=tracker_id,
                        )
                    )

                # for box in results.boxes:
                #     coordinates = box.xyxyn.cpu().numpy().flatten()
                #     x1_n, y1_n, x2_n, y2_n = coordinates[:4]
                #     boxes.append(
                #         player_inference_pb2.BoundingBox(
                #             x1_n=x1_n,
                #             y1_n=y1_n,
                #             x2_n=x2_n,
                #             y2_n=y2_n,
                #             confidence=box.conf.item(),
                #             class_label=labels[int(box.cls.item())],
                #         )
                # )
                logger.info(f"Frame ID {frame.frame_id} processed with {len(boxes)} detections.")

                yield player_inference_pb2.PlayerInferenceResponse(
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
