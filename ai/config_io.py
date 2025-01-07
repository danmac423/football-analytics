import json

from pathlib import Path
from typing import Any


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
