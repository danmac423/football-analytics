import cv2
import numpy as np
import supervision as sv

from football_analytics.football_pitch.draw_football_pitch import (
    draw_football_pitch,
    draw_points_on_pitch,
)
from football_analytics.football_pitch.football_pitch_configuration import (
    FootballPitchConfiguration,
)
from football_analytics.football_pitch.view_transformer import ViewTransformer

GOALKEEPER_ID = 0
PLAYER_ID = 1
REFEREE_ID = 2

def generate_radar(
        frame: np.ndarray,
        people_detections: sv.Detections,
        ball_detections: sv.Detections,
        keypoints_detections: sv.KeyPoints,

    ):
    height, width, _ = frame.shape

    CONFIG = FootballPitchConfiguration()

    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    if people_detections.class_id is not None:
        players_detections = people_detections[people_detections.class_id == PLAYER_ID]
        goalkeepers_detections = people_detections[people_detections.class_id == GOALKEEPER_ID]
        referees_detections = people_detections[people_detections.class_id == REFEREE_ID]
    else:
        players_detections = sv.Detections(
            xyxy=np.empty((0, 4))
        )
        goalkeepers_detections = sv.Detections(
            xyxy=np.empty((0, 4))
        )
        referees_detections = sv.Detections(
            xyxy=np.empty((0, 4))
        )

    filter = keypoints_detections.confidence[0] > 0.5
    frame_reference_points = keypoints_detections.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    view_transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = view_transformer.transform_points(frame_ball_xy)

    frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = view_transformer.transform_points(frame_players_xy)

    frame_goalkeepers_xy = goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_goalkeepers_xy = view_transformer.transform_points(frame_goalkeepers_xy)

    frame_referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_referees_xy = view_transformer.transform_points(frame_referees_xy)

    pitch = draw_football_pitch(config=CONFIG)


    if pitch_ball_xy.size != 0:
        pitch = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_ball_xy,
            main_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=pitch
        )

    if pitch_players_xy.size != 0:
        pitch = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy,
            main_color=sv.Color.RED,
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=pitch
        )

    if pitch_goalkeepers_xy.size != 0:
        pitch = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_goalkeepers_xy,
            main_color=sv.Color.BLUE,
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=pitch
        )

    if pitch_referees_xy.size != 0:
        pitch = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_referees_xy,
            main_color=sv.Color.YELLOW,
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=pitch
        )

    small_pitch_height = int(height * 0.2)
    small_pitch_width = int(pitch.shape[1] * (small_pitch_height / pitch.shape[0]))
    small_pitch = cv2.resize(pitch, (small_pitch_width, small_pitch_height))

    y_offset = height - small_pitch_height
    x_offset = (width - small_pitch_width) // 2
    frame[y_offset:height, x_offset:x_offset + small_pitch_width] = small_pitch


    return frame