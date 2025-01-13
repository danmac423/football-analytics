from concurrent import futures
from typing import Any, Generator, Iterator

import cv2
import grpc
import numpy as np
import supervision as sv

from services.ball_inference.grpc_files import ball_inference_pb2, ball_inference_pb2_grpc
from services.config import (
    BALL_INFERENCE_SERVICE_ADDRESS,
    INFERENCE_MANAGER_SERVICE_ADDRESS,
    PLAYER_INFERENCE_SERVICE_ADDRESS,
)
from services.inference_manager.grpc_files import inference_manager_pb2, inference_manager_pb2_grpc
from services.player_inference.grpc_files import player_inference_pb2, player_inference_pb2_grpc

BALL_COLOR = "#FF1493"
PLAYER_COLORS = ["#00BFFF", "#FF6347", "#FFD700"]

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

sv.get_video_frames_generator


class InferenceManagerServiceServicer(inference_manager_pb2_grpc.InferenceManagerServiceServicer):
    def __init__(self):
        self.ball_stub = ball_inference_pb2_grpc.YOLOBallInferenceServiceStub(
            grpc.insecure_channel(BALL_INFERENCE_SERVICE_ADDRESS)
        )
        self.player_stub = player_inference_pb2_grpc.YOLOPlayerInferenceServiceStub(
            grpc.insecure_channel(PLAYER_INFERENCE_SERVICE_ADDRESS)
        )

    def ProcessFrames(
        self, request_iterator: Iterator[ball_inference_pb2.Frame], context: grpc.ServicerContext
    ) -> Generator[inference_manager_pb2.Frame, Any, Any]:
        for frame in request_iterator:
            ball_response: ball_inference_pb2.BallInferenceResponse = next(
                self.ball_stub.InferenceBall(iter([frame]))
            )
            player_response: player_inference_pb2.PlayerInferenceResponse = next(
                self.player_stub.InferencePlayers(iter([frame]))
            )

            player_boxes = player_response.boxes
            ball_boxes = ball_response.boxes

            frame_ndarray = cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR)
            height, width, _ = frame_ndarray.shape

            xyxy = []
            confidences = []
            class_ids = []

            class_ids_map = {"goalkeeper": 0, "player": 1, "referee": 2}

            for box in player_boxes:
                x1 = int(box.x1_n * width)
                y1 = int(box.y1_n * height)
                x2 = int(box.x2_n * width)
                y2 = int(box.y2_n * height)

                xyxy.append([x1, y1, x2, y2])
                confidences.append(box.confidence)
                class_ids.append(class_ids_map[box.class_label])

            xyxy_array = np.array(xyxy, dtype=np.float32)
            confidence_array = np.array(confidences, dtype=np.float32)
            class_id_array = np.array(class_ids, dtype=object)

            detections = sv.Detections(
                xyxy=xyxy_array, confidence=confidence_array, class_id=class_id_array
            )

            annotated_frame = ELLIPSE_ANNOTATOR.annotate(frame_ndarray, detections)

            xyxy = []
            confidences = []
            class_ids = []

            try:
                for box in ball_boxes:
                    x1 = int(box.x1_n * width)
                    y1 = int(box.y1_n * height)
                    x2 = int(box.x2_n * width)
                    y2 = int(box.y2_n * height)

                    xyxy.append([x1, y1, x2, y2])
                    confidences.append(box.confidence)
                    class_ids.append(0)

                xyxy_array = np.array(xyxy, dtype=np.float32)
                confidence_array = np.array(confidences, dtype=np.float32)
                class_id_array = np.array(class_ids, dtype=object)

                detections = sv.Detections(
                    xyxy=xyxy_array, confidence=confidence_array, class_id=class_id_array
                )

                annotated_frame = TRIANGLE_ANNOTATOR.annotate(annotated_frame, detections)
            except:
                pass

            _, frame_bytes = cv2.imencode(".jpg", annotated_frame)

            yield inference_manager_pb2.Frame(content=frame_bytes.tobytes())


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = InferenceManagerServiceServicer()

    inference_manager_pb2_grpc.add_InferenceManagerServiceServicer_to_server(servicer, server)

    server.add_insecure_port(INFERENCE_MANAGER_SERVICE_ADDRESS)
    print(f"Inference Manager started on {INFERENCE_MANAGER_SERVICE_ADDRESS}.")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
