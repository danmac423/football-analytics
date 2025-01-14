import numpy as np
import supervision as sv
from multimethod import multimethod

from services.ball_inference.grpc_files import ball_inference_pb2
from services.keypoints_detection.grpc_files import keypoints_detection_pb2
from services.player_inference.grpc_files import player_inference_pb2


@multimethod
def to_supervision(ball_response: ball_inference_pb2.BallInferenceResponse, frame_ndarray: np.ndarray) -> sv.Detections:
    ball_boxes = ball_response.boxes
    height, width, _ = frame_ndarray.shape


    xyxy = []
    confidences = []
    class_ids = []

    try:
        for box in ball_boxes:
            x1 = int(box.x1_n * width)
            y1 = int(box.y1_n * height)
            x2 = int(box.x2_n * width)
            y2 = int(box.y2_n * height)

            xyxy.append([x1, y1, x2, y2])
            confidences.append(box.confidence)
            class_ids.append(0)

        xyxy_array = np.array(xyxy, dtype=np.float32)
        confidence_array = np.array(confidences, dtype=np.float32)
        class_id_array = np.array(class_ids, dtype=object)

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




@multimethod
def to_supervision(
    keypoints_response: keypoints_detection_pb2.KeypointsDetectionResponse,
    frame_ndarray: np.ndarray
) -> sv.KeyPoints:
    pitch_keypoints = keypoints_response.keypoints
    height, width, _ = frame_ndarray.shape # Can be used to annotate pitch box


    kp_xy = []
    try:
        for kp in pitch_keypoints:
            kp_x = int(kp.x)
            kp_y = int(kp.y)
            confidence = kp.confidence

            kp_xy.append([(kp_x, kp_y)])

        kp_xy_array = np.array(kp_xy, dtype=np.float32).reshape(1, -1, 2)

        keypoints = sv.KeyPoints(xy=kp_xy_array)

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



@multimethod
def to_supervision(
    player_response: player_inference_pb2.PlayerInferenceResponse,
    frame_ndarray: np.ndarray
) -> sv.Detections:
    height, width, _ = frame_ndarray.shape

    player_boxes = player_response.boxes

    xyxy = []
    confidences = []
    class_ids = []

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

        xyxy_array = np.array(xyxy, dtype=np.float32)
        confidence_array = np.array(confidences, dtype=np.float32)
        class_id_array = np.array(class_ids, dtype=object)

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