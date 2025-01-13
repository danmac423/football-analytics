import torch

INFERENCE_MANAGER_SERVICE_ADDRESS = "localhost:50051"

BALL_INFERENCE_MODEL_PATH = "models/ball_inference.pt"
BALL_INFERENCE_SERVICE_ADDRESS = "localhost:50052"

PLAYER_INFERENCE_MODEL_PATH = "models/player_inference.pt"
PLAYER_INFERENCE_SERVICE_ADDRESS = "localhost:50053"


DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.mps.is_available():
    DEVICE = "mps"
