import typer

from pathlib import Path
from loguru import logger
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics


app = typer.Typer()


def validate(path_to_best: Path) -> DetMetrics:
    logger.info(f"Validating model from {path_to_best}")

    model = YOLO(path_to_best)
    metrics = model.val()

    return metrics


@app.command()
def main(models_to_validate: list[Path]):
    for model in models_to_validate:
        metrics = validate(model)

        logger.info(f"mAP50-95: {metrics.box.map}")
        logger.info(f"mAP50: {metrics.box.map50}")
        logger.info(f"mAP75: {metrics.box.map75}")
        logger.info(f"list of mAP50-95 for each category: {metrics.box.maps}")


if __name__ == "__main__":
    app()
