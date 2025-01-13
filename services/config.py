import torch

PLAYER_INFERENCE_MODEL_PATH="models/player_inference.pt"
DEVICE="cpu"
if torch.cuda.is_available():
    DEVICE="cuda"
elif torch.mps.is_available():
    DEVICE = "mps"
