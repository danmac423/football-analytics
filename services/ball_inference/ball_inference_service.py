from concurrent import futures
from typing import Any, Generator, Iterator

import cv2
import grpc
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from services.ball_inference.grpc_files import ball_inference_pb2, ball_inference_pb2_grpc
from services.config import BALL_INFERENCE_MODEL_PATH


class YOLOBallInferenceServiceServicer(ball_inference_pb2_grpc.YOLOBallInferenceServiceServicer):
    def __init__(self):
        self.model = YOLO(BALL_INFERENCE_MODEL_PATH)

    def InferenceBall(
        self, request_iterator: Iterator[ball_inference_pb2.Frame], context: grpc.ServicerContext
    ) -> Generator[ball_inference_pb2.BallInferenceResponse, Any, Any]:
        for frame in request_iterator:
            frame_image = cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR)

            result: Results = self.model.predict(frame_image)
            labels: list[str] = result.names

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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = YOLOBallInferenceServiceServicer()

    ball_inference_pb2_grpc.add_YOLOBallInferenceServiceServicer_to_server(servicer, server)

    server.add_insecure_port("[::]:50051")
    print("Server started on port 50051.")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
