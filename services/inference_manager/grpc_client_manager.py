"""Module for managing the gRPC clients for the inference services."""

import grpc

from services.ball_inference.grpc_files import ball_inference_pb2_grpc
from services.keypoints_detection.grpc_files import keypoints_detection_pb2_grpc
from services.player_inference.grpc_files import player_inference_pb2_grpc


class GrpcClientManager:
    """Class to manage the gRPC clients for the inference services."""

    def __init__(self, ball_address, player_address, keypoints_address):
        self.ball_channel = grpc.insecure_channel(ball_address)
        self.player_channel = grpc.insecure_channel(player_address)
        self.keypoints_channel = grpc.insecure_channel(keypoints_address)

        self.ball_stub = ball_inference_pb2_grpc.YOLOBallInferenceServiceStub(self.ball_channel)
        self.player_stub = player_inference_pb2_grpc.YOLOPlayerInferenceServiceStub(
            self.player_channel
        )
        self.keypoints_stub = keypoints_detection_pb2_grpc.YOLOKeypointsDetectionServiceStub(
            self.keypoints_channel
        )

    def close(self):
        """Method to close the gRPC channels."""
        self.ball_channel.close()
        self.player_channel.close()
        self.keypoints_channel.close()
