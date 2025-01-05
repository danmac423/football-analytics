from datetime import datetime
from pathlib import Path

import typer
from loguru import logger
from ultralytics import YOLO

from ai.config import RUNS_DIR
from ai.config_io import read_from_json


app = typer.Typer()


def train(config: dict):
    model_path = config.pop("model")

    logger.info(f"Loading model from {model_path}")
    model = YOLO(model_path)

    logger.info(f"Training model with configuration {config}")
    model.train(**config)


@app.command()
def main(training_config_path: Path):
    logger.info(f"Reading configuration from {training_config_path}")

    for config in read_from_json(training_config_path):
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Current timestamp: {current_timestamp}")

        if "project" not in config.keys():
            config["project"] = RUNS_DIR / f"{config["model"][:-3]}" / f"train_{current_timestamp}"

        logger.info(f"Saving training run to: {config['project']}")

        if "additional_dataset" in config.keys():
            additional_dataset = config.pop("additional_dataset")

            train(config)

            config["data"] = additional_dataset
            config["model"] = config["project"] / "train" / "weights" / "best.pt"

            train(config)
        else:
            train(config)


if __name__ == "__main__":
    app()
