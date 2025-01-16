from typing import Optional

import cv2
import numpy as np
import supervision as sv

from football_analytics.football_pitch.football_pitch_configuration import (
    FootballPitchConfiguration,
)


def draw_football_pitch(
    config: FootballPitchConfiguration,
    background_color: sv.Color = sv.Color.from_hex("#7EAF34"),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1,
) -> np.ndarray:
    """Draw a football pitch based on the provided configuration.

    Args:
        config (FootballPitchConfiguration): configuration of the football pitch.
        background_color (sv.Color, optional): Background color of the pitch. Defaults to #7EAF34.
        line_color (sv.Color, optional): Lines color. Defaults white.
        padding (int, optional): Football pitch padding. Defaults to 50.
        line_thickness (int, optional): Thickness of pitch lines. Defaults to 4.
        point_radius (int, optional): Radius of the point on the pitch. Defaults to 8.
        scale (float, optional): Scale of the pitch. Defaults to 0.1.

    Returns:
        np.ndarray: Image of the football pitch.
    """

    def scale_and_pad(point: tuple[float, float]) -> tuple[int, int]:
        """Scale and pad a point to fit the image dimensions."""
        return (int(point[0] * scale) + padding, int(point[1] * scale) + padding)

    def draw_line(image: np.ndarray, start: int, end: int):
        """Draw a line on the pitch image."""
        cv2.line(
            img=image,
            pt1=scale_and_pad(config.vertices[start - 1]),
            pt2=scale_and_pad(config.vertices[end - 1]),
            color=line_color.as_bgr(),
            thickness=line_thickness,
        )

    def draw_circle(
        image: np.ndarray, center: tuple[float, float], radius: float, fill: bool = False
    ):
        """Draw a circle on the pitch image."""
        cv2.circle(
            img=image,
            center=scale_and_pad(center),
            radius=int(radius * scale),
            color=line_color.as_bgr(),
            thickness=-1 if fill else line_thickness,
        )

    def draw_penalty_spots(image: np.ndarray):
        """Draw penalty spots on the pitch image."""
        penalty_spots = [
            (config.penalty_spot_distance, config.width / 2),
            (config.length - config.penalty_spot_distance, config.width / 2),
        ]
        for spot in penalty_spots:
            draw_circle(image, spot, point_radius / scale, fill=True)

    image_height = int(config.width * scale) + 2 * padding
    image_width = int(config.length * scale) + 2 * padding
    pitch_image = np.ones((image_height, image_width, 3), dtype=np.uint8)
    pitch_image[:] = background_color.as_bgr()

    for start, end in config.edges:
        draw_line(pitch_image, start, end)

    draw_circle(
        image=pitch_image,
        center=(config.length / 2, config.width / 2),
        radius=config.centre_circle_radius,
    )

    draw_penalty_spots(pitch_image)

    return pitch_image


def draw_points_on_pitch(
    config: FootballPitchConfiguration,
    xy: np.ndarray,
    main_color: sv.Color = sv.Color.WHITE,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draws points on a football pitch.

    Args:
        config (FootballPitchConfiguration): Configuration of the football pitch.
        xy (np.ndarray): Points to draw on the pitch.
        main_color (sv.Color, optional): Main color of the points. Defaults to WHITE.
        edge_color (sv.Color, optional): Edge color of the points. Defaults to BLACK.
        radius (int, optional): Radius of the points. Defaults to 10.
        thickness (int, optional): Thickness of the points. Defaults to 2.
        padding (int, optional): Padding of the pitch. Defaults to 50.
        scale (float, optional): Scale of the pitch. Defaults to 0.1.
        pitch (np.ndarray, optional): Image of the pitch. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with points drawn on it.
    """

    def scale_and_pad(point):
        """Scale and pad a point to fit the image dimensions."""
        return (int(point[0] * scale) + padding, int(point[1] * scale) + padding)

    def draw_single_point(image, point):
        """Draw a single point on the pitch image."""
        scaled_point = scale_and_pad(point)
        cv2.circle(
            img=image, center=scaled_point, radius=radius, color=main_color.as_bgr(), thickness=-1
        )
        cv2.circle(
            img=image,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness,
        )

    if pitch is None:
        pitch = draw_football_pitch(config=config, padding=padding, scale=scale)

    for point in xy:
        draw_single_point(pitch, point)

    return pitch
