from concurrent import futures
from typing import Any, Generator, Iterator

import cv2
import grpc
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from services.config import DEVICE, PLAYER_INFERENCE_MODEL_PATH
from services.player_inference.grpc_files import player_inference_pb2, player_inference_pb2_grpc


class YOLOPlayerInferenceServiceServicer(
    player_inference_pb2_grpc.YOLOPlayerInferenceServiceServicer
):
    """YOLOPlayerInferenceServiceServicer class to implement the gRPC service.

    Attributes:
        model (YOLO): YOLO model object
    """
    def __init__(self):
        self.model = YOLO(PLAYER_INFERENCE_MODEL_PATH).to(DEVICE)

    def InferencePlayers(
            self,
            request_iterator: Iterator[player_inference_pb2.Frame],
            context: grpc.ServicerContext
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
        for frame in request_iterator:
            frame_image = cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR)

            results: Results = self.model(frame_image)[0]
            labels = results.names

            boxes = []

            for box in results.boxes:
                coordinates = box.xyxyn.cpu().numpy().flatten()
                x1_n, y1_n, x2_n, y2_n = coordinates[:4]
                boxes.append(
                    player_inference_pb2.BoundingBox(
                        x1_n=x1_n,
                        y1_n=y1_n,
                        x2_n=x2_n,
                        y2_n=y2_n,
                        confidence=box.conf.item(),
                        class_label=labels[int(box.cls.item())]
                    )
                )

            yield player_inference_pb2.PlayerInferenceResponse(
                frame_id=frame.frame_id,
                boxes=boxes
            )


def serve():
    """
    Function to start the gRPC server for the YOLO Player Detection Service.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    player_inference_pb2_grpc.add_YOLOPlayerInferenceServiceServicer_to_server(
        YOLOPlayerInferenceServiceServicer(),
        server
    )
    server.add_insecure_port('[::]:50052')
    print("YOLO Player Detection Service is running on port 50052")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()