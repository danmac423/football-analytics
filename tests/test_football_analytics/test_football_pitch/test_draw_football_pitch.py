from unittest.mock import patch

import numpy as np

from football_analytics.football_pitch.draw_football_pitch import (
    draw_football_pitch,
    draw_points_on_pitch,
)
from football_analytics.football_pitch.football_pitch_configuration import (
    FootballPitchConfiguration,
)


def test_draw_football_pitch_default():
    config = FootballPitchConfiguration()
    pitch_image = draw_football_pitch(config)

    expected_width = int(config.width * 0.1) + 2 * 50
    expected_height = int(config.length * 0.1) + 2 * 50
    assert pitch_image.shape == (expected_width, expected_height, 3)

    assert np.any(pitch_image)


def test_draw_football_pitch_custom_padding():
    config = FootballPitchConfiguration()
    padding = 100

    pitch_image = draw_football_pitch(config, padding=padding)

    expected_width = int(config.width * 0.1) + 2 * padding
    expected_height = int(config.length * 0.1) + 2 * padding
    assert pitch_image.shape == (expected_width, expected_height, 3)


def test_draw_points_on_pitch():
    config = FootballPitchConfiguration()
    points = np.array([[1000, 2000], [3000, 4000]])
    pitch_image = draw_football_pitch(config)

    result_image = draw_points_on_pitch(config, points, pitch=pitch_image)

    assert np.any(result_image)


def test_draw_points_on_pitch_empty():
    config = FootballPitchConfiguration()
    points = np.array([])

    pitch_image = draw_football_pitch(config)
    result_image = draw_points_on_pitch(config, points, pitch=pitch_image)

    assert np.array_equal(result_image, pitch_image)


def test_draw_points_on_pitch_none():
    config = FootballPitchConfiguration()
    points = np.array([[1000, 2000], [3000, 4000]])

    with patch(
        "football_analytics.football_pitch.draw_football_pitch.draw_football_pitch"
    ) as mock_draw_pitch:
        mock_draw_pitch.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)

        result_image = draw_points_on_pitch(config, points, pitch=None)

        mock_draw_pitch.assert_called_once()

        assert np.any(result_image)

    # result_image = draw_points_on_pitch(config, points, pitch=None)

    # assert np.any(result_image)
    # assert draw_football_pitch.assert_any_call()
