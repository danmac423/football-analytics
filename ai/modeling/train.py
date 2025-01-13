import os
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


def remove_label():
    datasets_dirs = os.listdir(f"{HOME}/datasets")
    datasets_dirs = [os.path.abspath(dir) for dir in datasets_dirs]

    datasests_to_clear = ["test", "train", "valid"]

    for datasets_dir in datasets_dirs:
        for dataset_to_clear in datasests_to_clear:
            for file_name in os.listdir(f"{datasets_dir}/{dataset_to_clear}/labels"):
                file_path = f"{datasets_dir}/{dataset_to_clear}/labels/{file_name}"

                with open(file_path, "r+") as f:
                    lines = f.readlines()
                    f.seek(0)
                    f.truncate(0)
                    for line in lines:
                        if line.startswith("0"):
                            continue

                        processed_line = " ".join(
                            str(int(float(value)) - 1) if i == 0 else value
                            for i, value in enumerate(line.split())
                        )
                        f.write(processed_line + "\n")

    datasets_dirs = os.listdir(f"{HOME}/datasets")
    datasets_dirs = [os.path.abspath(dir) for dir in datasets_dirs]

    datasests_to_clear = ["test", "train", "valid"]

    for datasets_dir in datasets_dirs:
        for dataset_to_clear in datasests_to_clear:
            for file_name in os.listdir(f"{datasets_dir}/{dataset_to_clear}/labels"):
                file_path = f"{datasets_dir}/{dataset_to_clear}/labels/{file_name}"

                with open(file_path, "r+") as f:
                    lines = f.readlines()
                    f.seek(0)
                    f.truncate(0)
                    for line in lines:
                        if line.startswith("0"):
                            continue

                        processed_line = " ".join(
                            str(int(float(value)) - 1) if i == 0 else value
                            for i, value in enumerate(line.split())
                        )
                        f.write(processed_line + "\n")


@app.command()
def main(training_config_path: Path):
    logger.info(f"Reading configuration from {training_config_path}")

    for config in read_from_json(training_config_path):
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Current timestamp: {current_timestamp}")

        if "project" not in config.keys():
            config["project"] = RUNS_DIR / f"{config["model"][:-3]}" / f"train_{current_timestamp}"

        logger.info(f"Saving training run to: {config['project']}")

        train(config)


if __name__ == "__main__":
    app()
