from dataclasses import dataclass, field
from typing import List, Tuple, Union


@dataclass
class FootballPitchConfiguration:
    # Default dimensions (in cm)
    width: int = 6800
    length: int = 10500
    penalty_box_width: int = 4032
    penalty_box_length: int = 1650
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @property
    def vertices(self) -> List[Tuple[Union[int, float], Union[int, float]]]:
        """Calculate the vertices of the soccer pitch.

        Returns:
            List[Tuple[Union[int, float], Union[int, float]]]: List of pitch vertices.
        """
        return [
            # Left side
            (0, 0),  # 1
            (0, (self.width - self.penalty_box_width) / 2),  # 2
            (0, (self.width - self.goal_box_width) / 2),  # 3
            (0, (self.width + self.goal_box_width) / 2),  # 4
            (0, (self.width + self.penalty_box_width) / 2),  # 5
            (0, self.width),  # 6
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
            (self.penalty_spot_distance, self.width / 2),  # 9
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
            # Middle
            (self.length / 2, 0),  # 14
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
            (self.length / 2, self.width),  # 17
            # Right side
            (
                self.length - self.penalty_box_length,
                (self.width - self.penalty_box_width) / 2,
            ),  # 18
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 19
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 20
            (
                self.length - self.penalty_box_length,
                (self.width + self.penalty_box_width) / 2,
            ),  # 21
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 23
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 24
            (self.length, 0),  # 25
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26
            (self.length, (self.width - self.goal_box_width) / 2),  # 27
            (self.length, (self.width + self.goal_box_width) / 2),  # 28
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29
            (self.length, self.width),  # 30
            # Center circle
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32
        ]

    edges: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (7, 8),
            (10, 11),
            (11, 12),
            (12, 13),
            (14, 15),
            (15, 16),
            (16, 17),
            (18, 19),
            (19, 20),
            (20, 21),
            (23, 24),
            (25, 26),
            (26, 27),
            (27, 28),
            (28, 29),
            (29, 30),
            (1, 14),
            (2, 10),
            (3, 7),
            (4, 8),
            (5, 13),
            (6, 17),
            (14, 25),
            (18, 26),
            (23, 27),
            (24, 28),
            (21, 29),
            (17, 30),
        ]
    )

    def __post_init__(self):
        """Validate the dimensions of the pitch."""
        if self.width <= 0 or self.length <= 0:
            raise ValueError("Width and length must be positive integers.")
        if self.penalty_box_width > self.width or self.goal_box_width > self.width:
            raise ValueError("Box widths cannot exceed the pitch width.")
        if self.penalty_box_length > self.length or self.goal_box_length > self.length:
            raise ValueError("Box lengths cannot exceed the pitch length.")
