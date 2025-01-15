from collections import deque

import numpy as np
import supervision as sv


class BallTracker:
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]

    def reset(self):
        self.buffer.clear()