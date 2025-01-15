import cv2
import numpy as np
import pytest

from football_analytics.football_pitch.view_transformer import ViewTransformer


def test_view_transformer_initialization():
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])

    transformer = ViewTransformer(source, target)
    assert transformer.metrics is not None
    assert transformer.metrics.shape == (3, 3)


def test_view_transformer_initialization_empty_matrices():
    source = np.array([])
    target = np.array([])

    with pytest.raises(cv2.error):
        ViewTransformer(source, target)


def test_transform_points():
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    points = np.array([[0.5, 0.5], [0.75, 0.75]])

    transformer = ViewTransformer(source, target)

    transformed_points = transformer.transform_points(points)

    assert transformed_points.shape == (2, 2)
    assert np.all(transformed_points >= 0)


def test_transform_points_empty():
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    points = np.array([])

    transformer = ViewTransformer(source, target)

    transformed_points = transformer.transform_points(points)

    assert transformed_points.size == 0