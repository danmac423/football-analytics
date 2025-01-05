import json

from pathlib import Path


def write_to_json(path: Path, data: dict[any:any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def read_from_json(path: Path) -> list[dict[any:any]]:
    with open(path, "r") as file:
        data = json.load(file)

        if isinstance(data, dict):
            return [data]
        else:
            return data
