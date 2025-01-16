import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
import typer
from loguru import logger
from optuna import Trial

from ai.config import PROJ_ROOT
from ai.config_io import read_from_json, write_to_json
from ai.modeling.train import do_remove_ball_label, train
from ai.modeling.validate import validate

RESULTS_DIRECTORY = PROJ_ROOT / "ai/experiments/results/yolo11_optuna_hyperparameter_search"


app = typer.Typer()


def save_trials_to_json(
    trial: Trial, search: dict[str, Any], config: dict[str, Any], value: float
) -> None:
    output_path = search["experiment_dir"] / f"trials_{search["model"][:-3]}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        try:
            trials_data = read_from_json(output_path)
        except json.JSONDecodeError:
            trials_data = []
    else:
        trials_data = []

    config["model"] = search.get("model")
    config["value"] = value

    if "remove_ball_label" in search.keys():
        config["remove_ball_label"] = search.get("remove_ball_label")

    trials_data.append(config)
    write_to_json(output_path, trials_data)

    logger.info(f"Saved trial {trial.number} configuration to {output_path}")


def create_config(
        *,
        model: str | None = None,
        task: str | None = None,
        data: str | None = None,
        epochs: int | None = None,
        batch: float | int | None = None,
        imgsz: int | None = None,
        plots: bool | None = None,
        project: str | None = None,
        mosaic: float | None = None,
        remove_ball_label: bool | None = None,
) -> dict[str, Any]:

    config = {}

    if model is not None:
        config["model"] = model
    if task is not None:
        config["task"] = task
    if data is not None:
        config["data"] = data
    if epochs is not None:
        config["epochs"] = epochs
    if batch is not None:
        config["batch"] = batch
    if imgsz is not None:
        config["imgsz"] = imgsz
    if plots is not None:
        config["plots"] = plots
    if project is not None:
        config["project"] = project
    if mosaic is not None:
        config["mosaic"] = mosaic
    if remove_ball_label is not None:
        config["remove_ball_label"] = remove_ball_label

    return config


def objective(trial: Trial, search: dict[str, Any]) -> float:
    epochs = trial.suggest_int(
        "epochs", search["epochs_min"], search["epochs_max"], step=search["epochs_step"]
    )
    batch = trial.suggest_float("batch", search["batch_min"], search["batch_max"])
    imgsz = trial.suggest_int(
        "imgsz", search["imgsz_min"], search["imgsz_max"], step=search["imgsz_step"]
    )

    config = create_config(
        model=str(search.get("model")),
        task=search.get("task"),
        data=str(search.get("data")),
        epochs=epochs,
        batch=batch if batch < 1 else int(batch),
        imgsz=imgsz,
        plots=search.get("plots"),
        project=str(search.get("project")),
        mosaic=search.get("mosaic")
    )

    if "remove_ball_label" in search.keys():
        config = do_remove_ball_label(config, str(search.get("current_timestamp")))

    train(config)

    train_number = str(trial.number + 1) if trial.number >= 1 else ""

    best_file = search["project"] / f"train{train_number}/weights/best.pt"

    validation_config = create_config(
        model=best_file,
        task=config.get("task"),
        data=str(config.get("data")),
        project=str(config.get("project")),
    )

    metrics = validate(validation_config)
    map50_95 = metrics.box.map

    save_trials_to_json(trial, search, config, map50_95)

    return map50_95


def run_trial(search: dict[str, Any]) -> optuna.Study:
    start = time.time()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, search), n_trials=search["numer_of_trials"])

    end = time.time()
    duration = end - start

    logger.info(
        "\n Parameter Optimization took %0.2f seconds (%0.1f minutes)" % (duration, duration / 60)
    )

    return study

def get_best_config(search: dict[str, Any]) -> dict[str, Any]:
    study = run_trial(search)

    best_config = create_config(
        model=str(search.get("model")),
        task="detect",
        data=str(search.get("data")),
        plots=search.get("plots"),
        mosaic=search.get("mosaic"),
        remove_ball_label=search.get("remove_ball_label"),
    )

    best_config.update(study.best_params)
    best_config["batch"] = (
        best_config["batch"] if best_config["batch"] < 1 else int(best_config["batch"])
    )

    return best_config


def prepare_search_paths(search: dict[str, Any]) -> dict[str, Any]:
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search["current_timestamp"] = current_timestamp
    logger.info(f"Current timestamp: {current_timestamp}")

    runs_dir = (
            PROJ_ROOT
            / f"ai/experiments/runs/{search["model"][:-3]}/experiment_{current_timestamp}"
    )
    search["project"] = runs_dir

    experiment_dir = RESULTS_DIRECTORY / f"{current_timestamp}"
    search["experiment_dir"] = experiment_dir

    logger.info(f"Using experiment directory: {experiment_dir}")

    search["data"] = os.path.abspath(search["data"])

    return search


def save_best_configuration(search: dict[str, Any], best_config: dict[str, Any]) -> None:
    output_path = search["experiment_dir"] / f"best_config_{search["model"][:-3]}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_to_json(output_path, [best_config])
    logger.info(f"Saved best config to {output_path}")


@app.command()
def main(path_to_hyperparameters_search_config: Path):
    searches = read_from_json(path_to_hyperparameters_search_config)

    for search in searches:
        search = prepare_search_paths(search)

        best_config = get_best_config(search)
        logger.info(f"Found best configuration: {best_config}")

        save_best_configuration(search, best_config)


if __name__ == "__main__":
    app()
