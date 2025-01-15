import os
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from ai.config_io import read_from_json

app = typer.Typer()


def validate(config: dict[str, Any]) -> DetMetrics:
    model_path = config.pop("model")

    logger.info(f"Loading model from {model_path}")
    model = YOLO(model_path)

    logger.info(f"Validating model with configuration {config}")
    metrics = model.val(**config)

    return metrics


@app.command()
def main(validation_config_path: Path):
    logger.info(f"Reading configuration from {validation_config_path}")

    for config in read_from_json(validation_config_path):
        config["model"] = os.path.abspath(config["model"])

        if "data" in config.keys():
            config["data"] = os.path.abspath(config["data"])

        if "project" not in config.keys():
            config["project"] = Path(config["model"]).parents[2]

        metrics = validate(config)

        logger.info(f"mAP50-95: {metrics.box.map}")
        logger.info(f"mAP50: {metrics.box.map50}")
        logger.info(f"mAP75: {metrics.box.map75}")
        logger.info(f"list of mAP50-95 for each category: {metrics.box.maps}")


if __name__ == "__main__":
    app()
