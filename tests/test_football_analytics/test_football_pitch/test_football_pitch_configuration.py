import pytest

from football_analytics.football_pitch.football_pitch_configuration import (
    FootballPitchConfiguration,
)


def test_valid_configuration():
    config = FootballPitchConfiguration()
    assert config.width == 6800
    assert config.length == 10500
    assert len(config.vertices) == 32
    assert len(config.edges) > 0


def test_invalid_width_and_length():
    with pytest.raises(ValueError, match="Width and length must be positive integers."):
        FootballPitchConfiguration(width=-500, length=-1000)


def test_box_width_exceeds_pitch():
    with pytest.raises(ValueError, match="Box widths cannot exceed the pitch width."):
        FootballPitchConfiguration(penalty_box_width=7000)


def test_box_length_exceeds_pitch():
    with pytest.raises(ValueError, match="Box lengths cannot exceed the pitch length."):
        FootballPitchConfiguration(penalty_box_length=11000)


def test_vertices_calculation():
    config = FootballPitchConfiguration()
    vertices = config.vertices
    assert len(vertices) == 32
    assert vertices[0] == (0, 0)
    assert vertices[-1] == (config.length // 2 + config.centre_circle_radius, config.width // 2)


def test_edges_structure():
    config = FootballPitchConfiguration()
    edges = config.edges
    assert len(edges) > 0
    assert all(len(edge) == 2 for edge in edges)  # Every edge with two ends
