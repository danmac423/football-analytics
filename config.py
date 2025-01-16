import torch

INFERENCE_MANAGER_SERVICE_ADDRESS = "localhost:50051"

BALL_INFERENCE_MODEL_PATH = "models/ball_inference.pt"
BALL_INFERENCE_SERVICE_ADDRESS = "localhost:50052"

PLAYER_INFERENCE_MODEL_PATH = "models/player_inference.pt"
PLAYER_INFERENCE_SERVICE_ADDRESS = "localhost:50053"

KEYPOINTS_DETECTION_MODEL_PATH = "models/keypoints_detection.pt"
KEYPOINTS_DETECTION_SERVICE_ADDRESS = "localhost:50054"

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.mps.is_available():
    DEVICE = "mps"


BALL_COLOR = "#BC0E0E"
PLAYER_COLORS = ["#1D8097", "#C1A2E7", "#FFCD10"]
KEYPOINTS_COLOR = "#FF1493"

GOALKEEPER_ID = 0
PLAYER_ID = 1
REFEREE_ID = 2
