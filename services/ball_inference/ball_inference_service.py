from concurrent import futures
from typing import Any, Generator, Iterator

import cv2
import grpc
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from services.ball_inference.grpc_files import ball_inference_pb2, ball_inference_pb2_grpc
from services.config import BALL_INFERENCE_MODEL_PATH, BALL_INFERENCE_SERVICE_ADDRESS, DEVICE


class YOLOBallInferenceServiceServicer(ball_inference_pb2_grpc.YOLOBallInferenceServiceServicer):
    """
    Class that implements the YOLOBallInferenceServiceServicer from the gRPC generated files.

    Attributes:
        model: The YOLO model to be used for inference.
    """

    def __init__(self):
        self.model = YOLO(BALL_INFERENCE_MODEL_PATH).to(DEVICE)

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
            frame_image = cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR)

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

            yield ball_inference_pb2.BallInferenceResponse(frame_id=frame.frame_id, boxes=boxes)


def serve():
    """
    Function that starts the gRPC server.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = YOLOBallInferenceServiceServicer()

    ball_inference_pb2_grpc.add_YOLOBallInferenceServiceServicer_to_server(servicer, server)

    server.add_insecure_port(BALL_INFERENCE_SERVICE_ADDRESS)
    print(f"Server started on {BALL_INFERENCE_SERVICE_ADDRESS}.")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
