import time
from datetime import datetime

import optuna
import json
import typer

from optuna import Trial
from pathlib import Path
from loguru import logger
from typing import Any

from football_analytics.config import PROJ_ROOT
from football_analytics.config_io import write_to_json, read_from_json
from football_analytics.modeling.train import train
from football_analytics.modeling.validate import validate


RESULTS_DIRECTORY = PROJ_ROOT / f"football_analytics/experiments/results/yolo11_optuna_hyperparameter_search"


app = typer.Typer()


def save_trials_to_json(trial: Trial, search: dict[str, Any], config: dict[str, Any], value: float) -> None:
    output_path = search["experiment_dir"] / f"trials_{search["model"][:-3]}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        try:
            trials_data = read_from_json(output_path)
        except json.JSONDecodeError:
            trials_data = []
    else:
        trials_data = []

    config["model"] = search["model"]
    config["value"] = value
    
    trials_data.append(config)
    write_to_json(output_path, trials_data)

    logger.info(f"Saved trial {trial.number} configuration to {output_path}")


def objective(trial: Trial, search: dict[str, Any]) -> float:
    epochs = trial.suggest_int("epochs", search["epochs_min"], search["epochs_max"], step=search["epochs_step"])
    batch = trial.suggest_float("batch", search["batch_min"], search["batch_max"])
    imgsz = trial.suggest_int('imgsz', search["imgsz_min"], search["imgsz_max"], step=search["imgsz_step"])

    config = {
        "model": str(search["model"]),
        "task": search["task"],
        "data": str(search["data"]),
        "epochs": epochs,
        "patience": search["patience"],
        "batch": batch if batch < 1 else int(batch),
        "imgsz": imgsz,
        "plots": True,
        "project": str(search["project"])
    }

    train(config)

    train_number = trial.number if trial.number >= 2 else ""
    best_file = search["project"] / f"train{train_number}/weights/best.pt"

    metrics = validate(best_file)
    map50_95 = metrics.box.map

    save_trials_to_json(trial, search, config, map50_95)

    return map50_95


def get_best_config(search: dict[str, Any]) -> dict[str, Any]:
    start = time.time()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, search), n_trials=search["numer_of_trials"])

    end = time.time()
    duration = end - start

    logger.info('\n Parameter Optimization took %0.2f seconds (%0.1f minutes)' % (duration, duration / 60))

    best_config = {
        "model": str(search["model"]),
        "task": "detect",
        "data": str(search["data"]),
        "plots": search["plots"],
        "patience": search["patience"],
    }

    best_config.update(study.best_params)
    best_config["batch"] = best_config["batch"] if best_config["batch"] < 1 else int(best_config["batch"])

    return best_config


@app.command()
def main(path_to_hyperparameters_search_config: Path):
    searches = read_from_json(path_to_hyperparameters_search_config)

    for search in searches:
        current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"Current timestamp: {current_timestamp}")

        runs_dir = PROJ_ROOT / f"football_analytics/experiments/runs/{search["model"][:-3]}/experiment_{current_timestamp}"
        search["project"] = runs_dir

        experiment_dir = RESULTS_DIRECTORY / f"{current_timestamp}"
        search["experiment_dir"] = experiment_dir

        logger.info(f"Using experiment directory: {experiment_dir}")

        best_config = get_best_config(search)
        logger.info(f"Found best configuration: {best_config}")

        output_path = search["experiment_dir"] / f"best_config_{search["model"][:-3]}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        write_to_json(output_path, best_config)
        logger.info(f"Saved best config to {output_path}")


if __name__ == "__main__":
    app()
