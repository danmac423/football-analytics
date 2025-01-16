from unittest.mock import patch

import numpy as np
import supervision as sv

from football_analytics.annotations.radar import generate_radar


@patch("football_analytics.camera_estimation.view_transformer.ViewTransformer")
@patch("football_analytics.football_pitch.draw_football_pitch.draw_football_pitch")
@patch("football_analytics.football_pitch.draw_football_pitch.draw_points_on_pitch")
def test_generate_radar_basic(mock_draw_points, mock_draw_pitch, mock_view_transformer):
    mock_draw_pitch.return_value = np.zeros((500, 300, 3), dtype=np.uint8)
    mock_draw_points.side_effect = lambda config, xy, main_color, edge_color, radius, pitch: pitch
    mock_view_transformer.return_value.transform_points.return_value = np.array([[50, 50]])

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    num_vertices = 32
    keypoints_xy = np.random.randint(0, 500, size=(1, num_vertices, 2))
    keypoints_confidence = np.random.uniform(0, 1, size=(1, num_vertices))

    keypoints_detections = sv.KeyPoints(
        xy=keypoints_xy,
        confidence=keypoints_confidence,
    )

    people_detections = sv.Detections(
        xyxy=np.array([[100, 100, 200, 200]]), class_id=np.array([1])
    )
    ball_detections = sv.Detections(xyxy=np.array([[300, 300, 310, 310]]))

    output_frame = generate_radar(frame, people_detections, ball_detections, keypoints_detections)

    assert isinstance(output_frame, np.ndarray)
    assert output_frame.shape == frame.shape
