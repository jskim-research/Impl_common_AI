"""Utility 함수 구현
"""
import os
from tensorflow.keras.models import Model
from pathlib import Path


def create_folder(path: str) -> None:
    """Create folder

    Args:
        path: folder path to create

    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Creating direction", path)


def save_model(model: Model, folder_name: str, file_name: str) -> None:
    """

    Args:
        model: model to save
        folder_name: (e.g., '../data/')
        file_name: (e.g., 'model_name.h5')

    """
    folder_path = "./" + folder_name
    file_path = "/" + file_name
    if os.path.isdir(folder_path):
        pass
    else:
        print("create new folder to save model")
        create_folder(folder_path)
    model.save(folder_path + file_path)


def load_subclass_model(model: Model, path: str) -> Model:
    """Load subclassed model

    Args:
        model: model that will load weights
        path: where model is saved

    Returns:
        loaded or not loaded model (if path does not exist, model can not be loaded)

    Note:
        subclassed model should use 'load_weights'.
    """
    print("trying to get data from", path)
    if os.path.exists(path):
        print(path, "found")
        # model one-shot forward (for model initilaiz_ath)
        print("load model")
        return model
    else:
        print(path, "not found")
        return model


def get_root_folder_path() -> Path:
    """Returns root folder path

    해당 파일의 경로를 root_folder/src/util.py 으로 가정하여 parent of parent를 반환함

    Return:
        root folder's path
    """
    return Path(__file__).absolute().parent.parent


if __name__ == "__main__":
    print(get_root_folder_path())
