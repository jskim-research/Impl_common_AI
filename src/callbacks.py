"""tensorflow callback 함수 모음

"""

import time
from tensorflow.keras import callbacks


def GetTensorboardCallback(path: str) -> callbacks.TensorBoard:
    """
    Args:
        path: tensorboard log directory (e.g., './tensor_board/model_name)

    Returns:
        Tensorboard 기록해주는 callback 함수
    """
    run_id = time.strftime("_%Y_%m_%d-%H_%M_%S")
    tb = callbacks.TensorBoard(log_dir=path + run_id)
    return tb




