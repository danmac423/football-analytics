import numpy as np

from football_analytics.camera_estimation.view_transformer import ViewTransformer
from football_analytics.utils.model import to_supervision
from services.player_inference.grpc_files import player_inference_pb2


class VelocityEstimator:
    """
    Class responsible for estimating the velocity of players.

    Attributes:
        previous_positions (dict): The previous positions of the players.
    """

    def __init__(self):
        self.previous_positions = {}

    def estimate_velocity(
        self,
        view_transformer: ViewTransformer,
        current_position: tuple[float, float],
        tracker_id: int,
        delta_time: float,
    ) -> float:
        """
        Estimates the velocity of a player. The velocity is calculated as the distance
        between the current position and the previous position divided by the time difference.

        Args:
            view_transformer (ViewTransformer): The view transformer.
            current_position (tuple[float, float]): The current position of the player.
            tracker_id (int): The tracker ID of the player.
            delta_time (float): The time difference between the current and previous frame.

        Returns:
            float: The estimated velocity of the player.
        """
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
    ) -> dict[int, float]:
        """
        Estimates the velocities of players. The velocities are calculated as the distance
        between the current position and the previous position divided by the time difference.

        Args:
            view_transformer (ViewTransformer): The view transformer.
            player_response (player_inference_pb2.PlayerInferenceResponse): The player
                inference response.
            frame_ndarray (np.ndarray): The frame in ndarray format.
            delta_time (float): The time difference between the current and previous frame.

        Returns:
            dict[int, float]: The estimated velocities of the players.
        """
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
