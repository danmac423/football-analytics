import argparse

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from requests import post
from ultralytics import YOLO

from football_analytics.utils.utils import (
    create_video_sink,
    list_to_nparray_in_dict,
    video_frames_generator,
)
from services.config import TRACKER_SERVICE_URL

# PITCH_KEYPOINTS_DETECTION_MODEL = YOLO("models/football-pitch-keypoints-detector-n.pt").to("mps")

CUSTOM_PALLETE = sv.ColorPalette(
    {
        0: sv.Color(255, 0, 0),  # ball
        1: sv.Color(0, 0, 255),  # goalkeeper
        2: sv.Color(0, 255, 0),  # player
        3: sv.Color(255, 255, 0),  # referee
    }
)

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(color=CUSTOM_PALLETE)


def run_player_detection(source_video_path: str, url=TRACKER_SERVICE_URL):
    """
    Performs player detection on video frames and annotates the detections.

    Processes a video frame by frame, runs object detection using a pre-trained YOLO model,
    and annotates the detected objects with ellipses. Uses a custom color palette to
    distinguish between different object classes. Yields annotated frames for further visualization
    or processing.

    Args:
        source_video_path (str): Path to the input video file to be processed.
        score_threshold (float, optional): The confidence threshold for detections.
            Detections with confidence lower than this value will be ignored.

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
        >>> for frame in run_player_detection(source_video_path):
        >>>     cv2.imshow("Annotated Frame", frame)
        >>>     if cv2.waitKey(1) & 0xFF == ord("q"):
        >>>         break
        >>> cv2.destroyAllWindows()
    """
    for frame in video_frames_generator(source_video_path):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        height, width, channels = frame.shape
        response = post(
            url,
            files={"file": frame},
            data={"height": height, "width": width, "channels": channels},
        ).json()

        response = list_to_nparray_in_dict(response)

        detections = sv.Detections(**response)

        image = ELLIPSE_ANNOTATOR.annotate(image, detections)

        annotated_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        yield annotated_frame


def run_pitch_keypoints_detection(source_video_path: str):
    """
    Detects and annotates pitch keypoints and bounding boxes in video frames.

    This function processes a video frame by frame, detects keypoints and bounding boxes
    using a pre-trained YOLO model, and annotates the detected features on the frames.
    Annotated frames are yielded for further visualization or processing.

    Args:
        source_video_path (str): Path to the input video file to be processed.

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
        >>> for frame in run_pitch_keypoints_detection(source_video_path):
        >>>     cv2.imshow("Annotated Frame", frame)
        >>>     if cv2.waitKey(1) & 0xFF == ord("q"):
        >>>         break
        >>> cv2.destroyAllWindows()
    """
    for frame in video_frames_generator(source_video_path):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # results = PITCH_KEYPOINTS_DETECTION_MODEL(image, mode="pose", imgsz=640)[0]
        results = YOLO("models/football-pitch-keypoints-detector-n.pt").to("mps")(
            image, mode="pose", imgsz=640
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        keypoints = sv.KeyPoints.from_ultralytics(results)

        image = sv.BoxAnnotator().annotate(image, detections)
        image = sv.VertexAnnotator().annotate(image, keypoints)
        # image = sv.EdgeAnnotator().annotate(image, keypoints)

        annotated_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        yield annotated_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_video_path",
        type=str,
        help="Path to the input video file",
    )
    parser.add_argument(
        "output_video_path",
        type=str,
        help="Path to the output video file",
    )

    args = parser.parse_args()
    source_video_path = args.source_video_path
    output_video_path = args.output_video_path

    out = create_video_sink(source_video_path, output_video_path)

    for frame in run_player_detection(source_video_path):
        out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")
