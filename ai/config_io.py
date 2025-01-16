import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


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

    datasets_to_clear = ["test", "train", "valid"]

    for dataset_to_clear in datasets_to_clear:
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


def copy_directory(src: str, dest: str):
    if not os.path.exists(src):
        raise ValueError(f"Source directory '{src}' does not exist.")

    try:
        shutil.copytree(src, dest)
        logger.info(f"Directory copied successfully from '{src}' to '{dest}'.")
    except FileExistsError:
        logger.info(f"Destination directory '{dest}' already exists.")
    except Exception as e:
        logger.info(f"An error occurred: {e}")
