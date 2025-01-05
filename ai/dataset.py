import shutil
from pathlib import Path

import kagglehub
import typer
from loguru import logger
from tqdm import tqdm

from ai.config import KAGGLE_DATASETS

app = typer.Typer()


@app.command()
def main():
    for dataset in tqdm(KAGGLE_DATASETS):
        kaggle_id = dataset["id"]
        save_dir = dataset["path"]

        logger.info(f"Downloading dataset with id {kaggle_id} from kaggle...")
        path = kagglehub.dataset_download(handle=kaggle_id)

        for item in Path(path).iterdir():
            logger.info(item)
            shutil.move(str(item), save_dir)

        logger.info(f"Dataset saved in {save_dir}")


if __name__ == "__main__":
    app()
