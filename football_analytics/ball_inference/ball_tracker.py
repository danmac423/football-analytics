"""Module for tracking the ball in a video sequence."""

from collections import deque

import numpy as np
import supervision as sv


class BallTracker:
    """
    Class to track the ball in a video sequence.

    Attributes:
        buffer (deque): The buffer to store the ball coordinates.
    """

    def __init__(self, buffer_size: int = 10):
        self.buffer: deque = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the tracker with the given detections. The tracker stores the ball coordinates
        in a buffer and returns the detection that is closest to the centroid of the buffer.

        Args:
            detections (sv.Detections): The detections.

        Returns:
            sv.Detections: The updated detections.
        """
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]  # type: ignore

    def reset(self):
        """Resets the tracker."""
        self.buffer.clear()
