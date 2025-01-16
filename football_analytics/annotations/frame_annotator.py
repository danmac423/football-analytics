
import cv2
import numpy as np
import supervision as sv

from config import BALL_COLOR, KEYPOINTS_COLOR, PLAYER_COLORS
from football_analytics.annotations.radar import generate_radar
from football_analytics.utils.model import to_supervision
from services.ball_inference.grpc_files import ball_inference_pb2
from services.keypoints_detection.grpc_files import keypoints_detection_pb2
from services.player_inference.grpc_files import player_inference_pb2


class FrameAnnotator:
    def __init__(self):
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(PLAYER_COLORS), thickness=2
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex(BALL_COLOR), base=20, height=15
        )
        self.vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex(KEYPOINTS_COLOR))

    def annotate_frame(
        self,
        frame_ndarray: np.ndarray,
        player_response: player_inference_pb2.PlayerInferenceResponse,
        ball_response: ball_inference_pb2.BallInferenceResponse,
        keypoints_response: keypoints_detection_pb2.KeypointsDetectionResponse,
        velocities: dict,
    ) -> np.ndarray:
        """
        Annotates a frame with player, ball, and keypoints data if available.
        """

        annotated_frame = frame_ndarray
        try:
            if player_response is not None:
                detections = to_supervision(player_response, frame_ndarray)
                annotated_frame = self.ellipse_annotator.annotate(annotated_frame, detections)

                for i, detection in enumerate(detections.xyxy):
                    tracker_id = int(detections.tracker_id[i])

                    velocity = velocities.get(tracker_id, 0.0)
                    velocity_text = f"{velocity:.2f} m/s"
                    position = (int(detection[0]), int(detection[1]))
                    cv2.putText(
                        annotated_frame,
                        velocity_text,
                        position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
        except Exception as e:
            raise Exception(f"Error annotating players: {e}")

        try:
            if ball_response is not None:
                detections = to_supervision(ball_response, frame_ndarray)
                annotated_frame = self.triangle_annotator.annotate(annotated_frame, detections)
        except Exception as e:
            raise Exception(f"Error annotating ball: {e}")

        try:
            if keypoints_response is not None:
                keypoints = to_supervision(keypoints_response, frame_ndarray)
                annotated_frame = self.vertex_annotator.annotate(annotated_frame, keypoints)
        except Exception as e:
            raise Exception(f"Error annotating keypoints: {e}")

        return annotated_frame

    @staticmethod
    def generate_radar(
        frame: np.ndarray,
        player_response: player_inference_pb2.PlayerInferenceResponse,
        ball_response: ball_inference_pb2.BallInferenceResponse,
        keypoints_response: keypoints_detection_pb2.KeypointsDetectionResponse,
    ):
        if not keypoints_response or not keypoints_response.keypoints:
            return frame

        if not player_response:
            player_detections = sv.Detections(
                xyxy=np.empty((0, 4)),
            )
        else:
            player_detections = to_supervision(player_response, frame)

        if not ball_response:
            ball_detections = sv.Detections(xyxy=np.empty((0, 4)))
        else:
            ball_detections = to_supervision(ball_response, frame)

        return generate_radar(
            frame, player_detections, ball_detections, to_supervision(keypoints_response, frame)
        )
