from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RUNS_DIR = PROJ_ROOT / "runs"

# Kaggle sources
KAGGLE_DATASETS = [
    {
        "id": "danielmachniak/football-players-detection",
        "path": RAW_DATA_DIR / "football-players-detection",
    },
    {
        "id": "danielmachniak/football-pitch-keypoints-detection",
        "path": RAW_DATA_DIR / "football-pitch-keypoints-detection",
    },
    {
        "id": "dominikaboguszewska/football-ball-detection",
        "path": RAW_DATA_DIR / "football-ball-detection",
    },
]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
