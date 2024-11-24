import cv2
import os
from typing import Any, Union, Dict, List, Tuple
import numpy as np


def video_frames_generator(video_path):
    """
    Generator which returns next video frames

    Args:
        video_path (string): path to video file

    Yield:
        Video frame in OpenCV fromat
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Plik wideo nie istnieje: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def serialize_object(
    obj: Any,
) -> Union[Dict[str, Any], List[Any], Tuple[Any, ...], str, float, int, None]:
    """
    Serializes an object into a JSON-compatible format.

    Args:
        obj (Any):
            The object to serialize. It can be:
            - Objects with attribute ("__dict__"
            - Dictionaries, lists, or tuples.
            - NumPy arrays (`np.ndarray`).
            - Basic types such as strings, integers, floats, or `None`.

    Returns:
        Union[Dict[str, Any], List[Any], Tuple[Any, ...], str, float, int, None]:
            A JSON-compatible representation of the object:
            - For custom objects, a dictionary of their attributes.
            - For dictionaries, recursively serialized key-value pairs.
            - For lists or tuples, recursively serialized elements.
            - For NumPy arrays, a Python list.
            - For basic types, the object itself.
            - For unsupported types, the string representation of the object.
    """
    if hasattr(obj, "__dict__"):
        return {k: serialize_object(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        return {k: serialize_object(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_object(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (str, int, float, type(None))):
        return obj
    else:
        return str(obj)
