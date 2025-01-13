import torch

PLAYER_INFERENCE_MODEL_PATH="models/player_inference.pt"

KEYPOINTS_DETECTION_MODEL_PATH="models/keypoints_detection.pt"
KEYPOINTS_DETECTION_SERVICE_ADDRESS="localhost:50054"


DEVICE="cpu"
if torch.cuda.is_available():
    DEVICE="cuda"
elif torch.mps.is_available():
    DEVICE = "mps"
