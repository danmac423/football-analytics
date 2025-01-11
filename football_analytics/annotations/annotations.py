

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from requests import post

from football_analytics.utils.utils import list_to_nparray_in_dict
from services.config import PITCH_KEYPOINTS_SERVICE_URL, TRACKER_SERVICE_URL

# CUSTOM_PALLETE = sv.ColorPalette(
#     {
#         0: sv.Color(255, 0, 0),  # ball
#         1: sv.Color(0, 0, 255),  # goalkeeper
#         2: sv.Color(0, 255, 0),  # player
#         3: sv.Color(255, 255, 0),  # referee
#     }
# )

CUSTOM_PALLETE = sv.ColorPalette(
    [
        sv.Color(255, 0, 0),  # ball
        sv.Color(0, 0, 255),  # goalkeeper
        sv.Color(0, 255, 0),  # player
        sv.Color(255, 255, 0),  # referee
    ]
)

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(color=CUSTOM_PALLETE)


def run_player_detection(frame: np.ndarray, url=TRACKER_SERVICE_URL):
    """
    Performs player detection on video frames and annotates the detections.

    Processes a video frame, runs object detection using a pre-trained YOLO model,
    and annotates the detected objects with ellipses. Uses a custom color palette to
    distinguish between different object classes. Yields annotated frames for further visualization
    or processing.

    Args:
        frame (np.ndarray): A single video frame in BGR format (compatible with OpenCV).
        url (str): URL of the player detection service.

    Yields:
        np.ndarray: Annotated video frame in BGR format (compatible with OpenCV), where
            detected objects are highlighted with ellipses.

    Notes:
        - A pre-trained YOLO model is used for object detection.
        - Detected objects are annotated with a `supervision.EllipseAnnotator` using a
          custom color palette.
        - The custom color palette maps classes (e.g., ball, player, referee) to specific colors.
        - The function does not write results to disk; it yields annotated frames for further use.

    Example:
        >>> source_video_path = "path/to/video.mp4"
        >>> for frame in video_frames_generator(source_video_path):
        >>>     frame_with_annotations = next(run_player_detection(frame))
        >>>     cv2.imshow("Annotated Frame", frame_with_annotations)
        >>>     if cv2.waitKey(1) & 0xFF == ord("q"):
        >>>         break
    """
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    height, width, channels = frame.shape


    response = post(
        url,
        files={"file": frame},
        data={"height": height, "width": width, "channels": channels},
    )

    if response.status_code != 200:
        raise ValueError(f"Error from server: {response.status_code} {response.text}")

    try:
        response = response.json()
    except ValueError:
        raise ValueError("Response is not valid JSON.")

    if not response:
        print(response)
        raise ValueError("Response is empty or invalid.")

    response = list_to_nparray_in_dict(response)

    detections = sv.Detections(**response)

    image = ELLIPSE_ANNOTATOR.annotate(image, detections)

    annotated_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    yield annotated_frame


def run_pitch_keypoints_detection(frame: np.ndarray, url=PITCH_KEYPOINTS_SERVICE_URL):
    """
    Detects and annotates pitch keypoints and bounding boxes in video frames.

    This function processes a video frame, detects keypoints and bounding boxes
    using a pre-trained YOLO model, and annotates the detected features on the frames.
    Annotated frames are yielded for further visualization or processing.

    Args:
        frame (np.ndarray): A single video frame in BGR format (compatible with OpenCV).
        url (str): URL of the pitch keypoints detection service.

    Yields:
        np.ndarray: Annotated video frame in BGR format (compatible with OpenCV), where
            detected bounding boxes and keypoints are highlighted.

    Notes:
        - The function uses a pre-trained YOLO model with "pose" mode for keypoint detection.
        - Bounding boxes are drawn using `supervision.BoxAnnotator`.
        - Keypoints are drawn using `supervision.VertexAnnotator`.
        - Detected features are visualized on each frame before yielding.

    Example:
        >>> source_video_path = "path/to/video.mp4"
        >>> for frame in video_frames_generator(source_video_path):
        >>>     frame_with_annotations = next(run_pitch_keypoints_detection(frame))
        >>>     cv2.imshow("Annotated Frame", frame_with_annotations)
        >>>     if cv2.waitKey(1) & 0xFF == ord("q"):
        >>>         break
    """
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    height, width, channels = frame.shape

    response = post(
        url,
        files={"file": frame},
        data={"height": height, "width": width, "channels": channels},
    )

    if response.status_code != 200:
        raise ValueError(f"Error from server: {response.status_code} {response.text}")

    try:
        response = response.json()
    except ValueError:
        raise ValueError("Response is not valid JSON.")

    if not response:
        print(response)
        raise ValueError("Response is empty or invalid.")


    detection_response = response["detections"]
    keypoints_response = response["keypoints"]

    detection_response = list_to_nparray_in_dict(detection_response)
    keypoints_response = list_to_nparray_in_dict(keypoints_response)

    if not detection_response:
        print("No detections in this frame.")
        yield frame

    detections = sv.Detections(**detection_response)
    keypoints = sv.KeyPoints(**keypoints_response)

    image = sv.BoxAnnotator().annotate(image, detections)
    image = sv.VertexAnnotator().annotate(image, keypoints)
    # image = sv.EdgeAnnotator().annotate(image, keypoints)

    annotated_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    yield annotated_frame
