"""Module to convert the response from the gRPC server to the supervision format."""

import numpy as np
import supervision as sv
from multimethod import multimethod

from services.ball_inference.grpc_files import ball_inference_pb2
from services.keypoints_detection.grpc_files import keypoints_detection_pb2
from services.player_inference.grpc_files import player_inference_pb2


@multimethod
def to_supervision(
    ball_response: ball_inference_pb2.BallInferenceResponse, frame_ndarray: np.ndarray
) -> sv.Detections:
    """
    Convert the response from the ball inference service to the supervision format.

    Args:
        ball_response (ball_inference_pb2.BallInferenceResponse): Response from the ball inference
            service
        frame_ndarray (np.ndarray): Frame in ndarray format

        Returns:
            sv.Detections: Detections in the supervision format
    """
    ball_boxes = ball_response.boxes
    height, width, _ = frame_ndarray.shape

    xyxy = []
    confidences = []
    class_ids = []

    try:
        for box in ball_boxes:
            if (
                not (0 <= box.x1_n <= 1)
                or not (0 <= box.y1_n <= 1)
                or not (0 <= box.x2_n <= 1)
                or not (0 <= box.y2_n <= 1)
            ):
                raise ValueError(f"Bounding box coordinates must be in [0, 1], got {box}")

            x1 = int(box.x1_n * width)
            y1 = int(box.y1_n * height)
            x2 = int(box.x2_n * width)
            y2 = int(box.y2_n * height)

            xyxy.append([x1, y1, x2, y2])
            confidences.append(box.confidence)
            class_ids.append(0)

        xyxy_array = np.array(xyxy, dtype=np.float32) if xyxy else np.empty((0, 4))
        confidence_array = np.array(confidences, dtype=np.float32) if confidences else None
        class_id_array = np.array(class_ids, dtype=object) if class_ids else None

        detections = sv.Detections(
            xyxy=xyxy_array, confidence=confidence_array, class_id=class_id_array
        )

        return detections

    except ValueError as ve:
        print(f"ValueError occurred: {ve}")
        raise
    except AttributeError as ae:
        print(f"AttributeError occurred: {ae}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


@multimethod  # type: ignore
def to_supervision(  # noqa
    keypoints_response: keypoints_detection_pb2.KeypointsDetectionResponse,
    frame_ndarray: np.ndarray,
) -> sv.KeyPoints:
    pitch_keypoints = keypoints_response.keypoints
    height, width, _ = frame_ndarray.shape  # Can be used to annotate pitch box

    kp_xy = []
    kp_conf = []
    try:
        for kp in pitch_keypoints:
            kp_x = int(kp.x)
            kp_y = int(kp.y)
            confidence = kp.confidence

            kp_xy.append([(kp_x, kp_y)])
            kp_conf.append(confidence)

        kp_xy_array = (
            np.array(kp_xy, dtype=np.float32).reshape(1, -1, 2) if kp_xy else np.empty((0, 0, 2))
        )
        kp_conf_array = np.array(kp_conf, dtype=np.float32).reshape(1, -1) if kp_conf else None

        keypoints = sv.KeyPoints(xy=kp_xy_array, confidence=kp_conf_array)

        return keypoints

    except ValueError as ve:
        print(f"ValueError occurred: {ve}")
        raise
    except AttributeError as ae:
        print(f"AttributeError occurred: {ae}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


@multimethod  # type: ignore
def to_supervision(  # noqa
    player_response: player_inference_pb2.PlayerInferenceResponse, frame_ndarray: np.ndarray
) -> sv.Detections:
    height, width, _ = frame_ndarray.shape

    player_boxes = player_response.boxes

    xyxy = []
    confidences = []
    class_ids = []
    tracker_ids = []

    class_ids_map = {"goalkeeper": 0, "player": 1, "referee": 2}

    try:
        for box in player_boxes:
            x1 = int(box.x1_n * width)
            y1 = int(box.y1_n * height)
            x2 = int(box.x2_n * width)
            y2 = int(box.y2_n * height)

            xyxy.append([x1, y1, x2, y2])
            confidences.append(box.confidence)
            class_ids.append(class_ids_map[box.class_label])
            tracker_ids.append(box.tracker_id)

        xyxy_array = (
            np.array(
                xyxy,
                dtype=np.float32,
            )
            if xyxy
            else np.empty((0, 4))
        )
        confidence_array = np.array(confidences, dtype=np.float32) if confidences else None
        class_id_array = np.array(class_ids, dtype=object) if class_ids else None
        tracker_id_array = np.array(tracker_ids, dtype=np.int32) if tracker_ids else None

        detections = sv.Detections(
            xyxy=xyxy_array,
            confidence=confidence_array,
            class_id=class_id_array,
            tracker_id=tracker_id_array,
        )

        return detections

    except ValueError as ve:
        print(f"ValueError occurred: {ve}")
        raise
    except AttributeError as ae:
        print(f"AttributeError occurred: {ae}")
        raise
    except KeyError as ke:
        print(f"KeyError occured: {ke}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
