import os
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from ultralytics import YOLO

from ai.config import RUNS_DIR
from ai.config_io import (
    copy_directory,
    read_from_json,
    remove_ball_label_from_data_yaml,
    remove_label_zero,
)

app = typer.Typer()


def train(config: dict):
    model_path = config.pop("model")

    logger.info(f"Loading model from {model_path}")
    model = YOLO(model_path)

    logger.info(f"Training model with configuration {config}")
    model.train(**config)


def do_remove_ball_label(config: dict[str, Any], current_timestamp: str) -> dict[str, Any]:
    dataset_directory = config["data"]

    if os.path.isfile(dataset_directory):
        data_file = os.path.basename(dataset_directory)
        dataset_directory = os.path.dirname(dataset_directory)
        add_data_file = True
    else:
        add_data_file = False

    dataset_copy = dataset_directory + f"-copy-{current_timestamp}"

    if add_data_file:
        config["data"] = dataset_copy + f"/{data_file}"

    copy_directory(dataset_directory, dataset_copy)

    logger.info("Removing ball label")
    remove_ball_label_from_data_yaml(config["data"])
    remove_label_zero(config["data"])

    return config


@app.command()
def main(training_config_path: Path):
    logger.info(f"Reading configuration from {training_config_path}")

    for config in read_from_json(training_config_path):
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Current timestamp: {current_timestamp}")

        config["data"] = os.path.abspath(config["data"])

        if "project" not in config.keys():
            config["project"] = RUNS_DIR / f"{config['model'][:-3]}" / f"train_{current_timestamp}"

        logger.info(f"Saving training run to: {config['project']}")

        if "remove_ball_label" in config.keys():
            config = do_remove_ball_label(config, current_timestamp)
            config.pop("remove_ball_label")

        train(config)


if __name__ == "__main__":
    app()
