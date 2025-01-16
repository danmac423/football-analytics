import numpy as np

from football_analytics.camera_estimation.view_transformer import ViewTransformer
from football_analytics.utils.model import to_supervision
from services.player_inference.grpc_files import player_inference_pb2


class VelocityEstimator:
    def __init__(self):
        self.previous_positions = {}

    def estimate_velocity(
        self,
        view_transformer: ViewTransformer,
        current_position: tuple[float, float],
        tracker_id: int,
        delta_time: float,
    ):
        real_position = view_transformer.transform_points(np.array([current_position]))[0]

        if tracker_id in self.previous_positions:
            prev_position = self.previous_positions[tracker_id]

            prev_real_position = view_transformer.transform_points(np.array([prev_position]))[0]

            distance = np.sqrt(
                (real_position[0] - prev_real_position[0]) ** 2
                + (real_position[1] - prev_real_position[1]) ** 2
            )

            velocity = (distance / 100.0) / delta_time if delta_time > 0 else 0
        else:
            velocity = 0

        self.previous_positions[tracker_id] = current_position
        return velocity

    def estimate_velocities(
        self,
        view_transformer: ViewTransformer,
        player_response: player_inference_pb2.PlayerInferenceResponse,
        frame_ndarray: np.ndarray,
        delta_time: float,
    ):
        try:
            if player_response is not None:
                velocities = {}

                detections = to_supervision(player_response, frame_ndarray)

                for i, detection in enumerate(detections.xyxy):
                    tracker_id = int(detections.tracker_id[i])
                    position = (
                        (detection[0] + detection[2]) / 2,  # x1 + x2 / 2
                        (detection[1] + detection[3]) / 2,  # y1 + y2 / 2
                    )
                    velocities[tracker_id] = self.estimate_velocity(
                        view_transformer, position, tracker_id, delta_time
                    )

                return velocities
        except Exception as e:
            raise Exception(f"Error annotating players: {e}")
