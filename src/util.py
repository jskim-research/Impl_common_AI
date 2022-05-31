"""Utility 함수 구현
"""
from pathlib import Path


def get_root_folder_path() -> Path:
    """Returns root folder path

    해당 파일의 경로를 root_folder/src/util.py 으로 가정하여 parent of parent를 반환함
    
    Return:
        root folder's path
    """
    return Path(__file__).absolute().parent.parent


if __name__ == "__main__":
    print(get_root_folder_path())
