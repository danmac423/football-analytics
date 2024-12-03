import argparse
from pathlib import Path

import typer
from loguru import logger
from ultralytics import YOLO

from football_analytics.config_io import read_from_json


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
        train(config)
        # print(config)


if __name__ == "__main__":
    app()

