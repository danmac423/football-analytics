import logging
import os
from concurrent import futures
from typing import Any, Generator, Iterator

import cv2
import grpc
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from services.config import (
	DEVICE,
	KEYPOINTS_DETECTION_MODEL_PATH,
	KEYPOINTS_DETECTION_SERVICE_ADDRESS,
)
from services.keypoints_detection.grpc_files import (
	keypoints_detection_pb2,
	keypoints_detection_pb2_grpc,
)

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/keypoints_detection_service.log")
    ]
)
logger = logging.getLogger(__name__)

class YOLOKeypointsDetectionServiceServicer(
	keypoints_detection_pb2_grpc.YOLOKeypointsDetectionServiceServicer
):
	"""YOLOKeypointsDetectionServiceServicer is a class that implements the
	YOLOKeypointsDetectionServiceServicer interface in the generated
	keypoints_detection_pb2_grpc module.

	Attributes:
		model (YOLO): An instance of the YOLO class from the ultralytics module.
	"""
	def __init__(self):
		logger.info("Initializing YOLO model...")
		self.model = YOLO(KEYPOINTS_DETECTION_MODEL_PATH).to(DEVICE)
		logger.info(f"YOLO model loaded from {KEYPOINTS_DETECTION_MODEL_PATH} on device {DEVICE}.")


	def DetectKeypoints(
			self,
            request_iterator: Iterator[keypoints_detection_pb2.Frame],
            context: grpc.ServicerContext
        ) -> Generator[keypoints_detection_pb2.KeypointsDetectionResponse, Any, Any]:
		"""DetectKeypoints method for the gRPC service which takes a stream of frames
		and returns the response with the bounding boxes and keypoints.

		Args:
			request_iterator (Iterator[keypoints_detection_pb2.Frame]): request iterator
			context (grpc.ServicerContext): context object for the request

		Yields:
			Generator[keypoints_detection_pb2.KeypointsDetectionResponse, Any, Any]: returns the response
			with frame_id, boxes (normalized), and keypoints
		"""
		for frame in request_iterator:
			try:
				frame_image = cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR)

				frame_image_resized = cv2.resize(frame_image, (640, 640))
				results: Results = self.model(frame_image_resized)[0]
				labels = results.names
				boxes = []
				keypoints = []

				original_height, original_width, _ = frame_image.shape
				height, width, _ = frame_image_resized.shape

				for box, keypoint in zip(results.boxes, results.keypoints):
					coordinates = box.xyxyn.cpu().numpy().flatten()
					x1_n, y1_n, x2_n, y2_n = coordinates[:4]
					boxes.append(
						keypoints_detection_pb2.BoundingBox(
							x1_n=x1_n,
							y1_n=y1_n,
							x2_n=x2_n,
							y2_n=y2_n,
							confidence=box.conf.item(),
							class_label=labels[int(box.cls.item())]
						)
					)

					# dziala
					# for point in keypoint.xyn[0].cpu().numpy():
					# 	x_n, y_n = point[:2]  # Znormalizowane współrzędne
					# 	print(x_n, y_n)
					# 	x = x_n * original_width
					# 	y = y_n * original_height
					# 	confidence = point[2] if len(point) > 2 else 0.0
					# 	keypoints.append(
					# 		keypoints_detection_pb2.Keypoint(
					# 			x=float(x),
					# 			y=float(y),
					# 			confidence=float(confidence)
					# 		)
					# 	)


					for point in keypoint.data.cpu().numpy()[0]:
						x, y = point[:2]
						x = x / width * original_width
						y = y / height * original_height
						confidence = point[2] if len(point) > 2 else 0.0
						keypoints.append(
							keypoints_detection_pb2.Keypoint(
								x=float(x),
								y=float(y),
								confidence=float(confidence)
							)
						)

				logger.info(
					f"Frame ID {frame.frame_id} processed with {len(keypoints)} keypoints and "
					f"{len(boxes)} detections."
				)

				yield keypoints_detection_pb2.KeypointsDetectionResponse(
					frame_id=frame.frame_id,
					boxes=boxes,
					keypoints=keypoints
				)
			except Exception as e:
				logger.error(f"Error processing frame ID {frame.frame_id}: {e}")
				context.abort(grpc.StatusCode.UNKNOWN, str(e))

def serve():
	"""
	Function that starts the gRPC server and adds the YOLOKeypointsDetectionServiceServicer
	"""
	logger.info("Starting gRPC server...")

	server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
	keypoints_detection_pb2_grpc.add_YOLOKeypointsDetectionServiceServicer_to_server(
		YOLOKeypointsDetectionServiceServicer(),
		server
	)
	server.add_insecure_port(KEYPOINTS_DETECTION_SERVICE_ADDRESS)
	logger.info(f"Server started on {KEYPOINTS_DETECTION_SERVICE_ADDRESS}.")
	server.start()
	server.wait_for_termination()

if __name__ == '__main__':
    serve()
