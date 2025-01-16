from unittest.mock import MagicMock, patch

import numpy as np

from football_analytics.keypoint_detection.yolo_keypoints_detector import YOLOKeypointsDetector
from services.keypoints_detection.grpc_files import keypoints_detection_pb2


@patch("football_analytics.keypoint_detection.yolo_keypoints_detector.YOLO")
def test_yolo_keypoints_detector_initialization(mock_yolo):
    mock_yolo.return_value = MagicMock()
    detector = YOLOKeypointsDetector()

    assert detector.model is not None


@patch("football_analytics.keypoint_detection.yolo_keypoints_detector.YOLO")
@patch("football_analytics.keypoint_detection.yolo_keypoints_detector.cv2.resize")
@patch("football_analytics.keypoint_detection.yolo_keypoints_detector.cv2.imdecode")
def test_detect_keypoints(mock_imdecode, mock_resize, mock_yolo):
    mock_model = MagicMock()
    mock_model.return_value = [MagicMock()]
    mock_yolo.return_value = mock_model
    mock_imdecode.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)

    detector = YOLOKeypointsDetector()
    detector._extract_boxes = MagicMock(
        return_value=[
            keypoints_detection_pb2.BoundingBox(
                x1_n=0.1, y1_n=0.2, x2_n=0.3, y2_n=0.4, confidence=0.9
            )
        ]
    )
    detector._extract_keypoints = MagicMock(
        return_value=[keypoints_detection_pb2.Keypoint(x=320, y=240, confidence=0.95)]
    )

    frame = keypoints_detection_pb2.Frame(frame_id=1, content=b"mock_frame")
    response = detector.detect_keypoints(frame)

    assert isinstance(response, keypoints_detection_pb2.KeypointsDetectionResponse)
    assert response.frame_id == 1
    assert len(response.boxes) == 1
    assert len(response.keypoints) == 1
    detector._extract_boxes.assert_called_once()
    detector._extract_keypoints.assert_called_once()


@patch("football_analytics.keypoint_detection.yolo_keypoints_detector.YOLO")
def test_extract_boxes(mock_yolo):
    mock_results = MagicMock()
    mock_results.names = {0: "person", 1: "ball"}
    mock_box = MagicMock()
    mock_box.xyxyn.cpu().numpy.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
    mock_box.conf.item.return_value = 0.9
    mock_box.cls.item.return_value = 1
    mock_results.boxes = [mock_box]

    detector = YOLOKeypointsDetector()
    boxes = detector._extract_boxes(mock_results)

    assert len(boxes) == 1
    assert np.isclose(boxes[0].confidence, 0.9)
    assert boxes[0].class_label == "ball"
    assert np.isclose(boxes[0].x1_n, 0.1)
    assert np.isclose(boxes[0].y1_n, 0.2)
    assert np.isclose(boxes[0].x2_n, 0.3)
    assert np.isclose(boxes[0].y2_n, 0.4)


@patch("football_analytics.keypoint_detection.yolo_keypoints_detector.YOLO")
def test_extract_keypoints(mock_yolo):
    mock_results = MagicMock()
    mock_keypoint = MagicMock()
    mock_keypoint.data.cpu().numpy.return_value = [[[240, 320, 0.9], [320, 240, 0.95]]]
    mock_results.keypoints = [mock_keypoint]

    detector = YOLOKeypointsDetector()
    keypoints = detector._extract_keypoints(
        mock_results,
        original_shape=(480, 640, 3),
        resized_shape=(640, 640, 3),
    )

    assert len(keypoints) == 2
    assert np.isclose(keypoints[0].confidence, 0.9)
    assert np.isclose(keypoints[1].confidence, 0.95)
    assert np.isclose(keypoints[0].x, 240)
    assert np.isclose(keypoints[0].y, 240)
    assert np.isclose(keypoints[1].x, 320)
    assert np.isclose(keypoints[1].y, 180)


@patch("football_analytics.keypoint_detection.yolo_keypoints_detector.cv2.imdecode")
def test_decode_frame(mock_imdecode):
    mock_imdecode.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    detector = YOLOKeypointsDetector()
    frame_image = detector._decode_frame(b"mock_frame")

    assert frame_image.shape == (480, 640, 3)
    mock_imdecode.assert_called_once()
