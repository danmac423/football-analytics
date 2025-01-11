import os
from typing import Any, Dict, List, Tuple, Union

import cv2
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


def list_to_nparray_in_dict(dictionary: dict) -> dict:
    """
    Converts lists from dictionary to NumPy arrays.
    """
    dictionary = dictionary.copy()
    for key, value in dictionary.items():
        if isinstance(value, list):
            dictionary[key] = np.array(value)
    return dictionary


def create_video_sink(source_video_path: str, output_video_path: str) -> cv2.VideoWriter:
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open the video file: {source_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
    sink = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    return sink
