import os
import shutil

import kagglehub


def clear_kagglehub_cache():
    """
    Clears the KaggleHub cache directory to ensure fresh downloads.
    """
    cache_dir = os.path.expanduser("~/.cache/kagglehub")
    if os.path.exists(cache_dir):
        print(f"Clearing KaggleHub cache at: {cache_dir}")
        shutil.rmtree(cache_dir)
    else:
        print("KaggleHub cache is already clean.")


def download_models_from_kaggle(model_ids: list[str], output_dir: str = "models"):
    """Download models from Kaggle.

    Args:
        model_ids (): _description_
        output_dir (_type_): _description_
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model_path in model_ids:
        print(f"Downloading dataset model from: {model_path}")

        path_to_files = kagglehub.model_download(model_path)
        print(f"Files downloaded to: {path_to_files}")

        if path_to_files and os.listdir(path_to_files):
            for file in os.listdir(path_to_files):
                source_path = os.path.join(path_to_files, file)
                target_path = os.path.join(output_dir, file)
                os.rename(source_path, target_path)

        print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    clear_kagglehub_cache()

    model_ids = [
        "danielmachniak/ball-inference-model/pyTorch/default",
        "danielmachniak/player-inference-model/pyTorch/default",
        "danielmachniak/keypoints-detection-model/pyTorch/default",
    ]

    download_models_from_kaggle(model_ids)

    print("Models saved to models directory.")
