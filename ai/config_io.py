import json
import os

from pathlib import Path
from typing import Any
from loguru import logger

import yaml


def write_to_json(path: Path, data: list[dict[Any, Any]]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def read_from_json(path: Path) -> list[dict[Any, Any]]:
    with open(path, "r") as file:
        data = json.load(file)

        if isinstance(data, dict):
            return [data]
        else:
            return data


def get_nc_from_data_yaml(file_path: str) -> int:
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    if 'nc' in data:
        return data['nc']
    else:
        raise KeyError(f"'nc' field is not found in the YAML file at {file_path}.")


def remove_ball_label_from_data_yaml(file_path: str):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    data['nc'] = 3

    if 'names' in data and 'ball' in data['names']:
        data['names'].remove('ball')

    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    logger.info(f"Successfully modified YAML file at {file_path}!")


def remove_label_zero(dataset_directory: str):
    if os.path.isfile(dataset_directory):
        dataset_directory = os.path.dirname(dataset_directory)

    dataset_to_clear = ["test", "train", "valid"]

    for dataset_to_clear in dataset_to_clear:
        for file_name in os.listdir(f"{dataset_directory}/{dataset_to_clear}/labels"):
            file_path = f"{dataset_directory}/{dataset_to_clear}/labels/{file_name}"

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
