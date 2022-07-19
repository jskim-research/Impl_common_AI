"""
Base model class 생성

Notes:
    모델 구현 시 항상 생각할 부분
    이렇게 구현할 경우 중간 output들이 shape와 동일한가? => padding은 추가했는가?
    activation은 추가했는가? (거의 relu)
    conv 다음에 pooling이 생략되지 않았는가?
    normalization은?

ToDo:
    * 예외 케이스 처리 (get_grad_cam input shape check 등)
    * grad_cam++ 구현 (https://velog.io/@tobigs_xai/CAM-Grad-CAM-Grad-CAMpp)
"""
import tensorflow.compat.v2 as tf
import copy
import itertools
import json
import os
import warnings
import weakref
import keras
from tensorflow.python.eager import context
from keras import backend
from keras import callbacks as callbacks_module
from keras import optimizer_v1
from keras import optimizers
from keras.engine import base_layer
from keras.engine import base_layer_utils
from keras.engine import compile_utils
from keras.engine import data_adapter
from keras.engine import training_utils
from keras.mixed_precision import loss_scale_optimizer as lso
from keras.mixed_precision import policy
from keras.saving import hdf5_format
from keras.saving import save
from keras.saving import saving_utils
from keras.saving import pickle_utils
from keras.saving.saved_model import json_utils
from keras.saving.saved_model import model_serialization
from keras.utils import generic_utils
from keras.utils import layer_utils
from keras.utils import object_identity
from keras.utils import tf_utils
from keras.utils import traceback_utils
from keras.utils import version_utils
from keras.utils.io_utils import ask_to_proceed_with_overwrite
from keras.utils.io_utils import path_to_string
from keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls


class BaseModel(keras.Model):
    """
    하나의 출력만 가지는 모델 생성
    다수의 출력을 가지는 경우 첫 번째 출력만 인정하도록 변경
    """
    def __init__(self, model: keras.Model):
        """
        Args:
            model: model composed of functional APIs
        """
        super().__init__()
        self.model = model

    def call(self, inputs, training=None, mask=None):
        """첫 번째 출력만 반환"""
        outputs = self.model(inputs)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            return outputs[0]
        else:
            return outputs

    def get_config(self):
        config = super().get_config()
        # 모델 정보 추가 가능
        return config


class RegressionGradCamModel(BaseModel):
    """
    CNN 모델이 입력 이미지에서 어느 부분을 중점적으로 봤는 지 계산 기능 추가\n
    대상 모델은 call 함수 후 [회귀 예측 결과, 마지막 convolution 결과]를 출력해야 한다\n
    Reference:
    - https://velog.io/@tobigs_xai/CAM-Grad-CAM-Grad-CAMpp
    """
    def __init__(self, model: keras.Model):
        super().__init__(model)

    def get_grad_cam(self, inputs):
        """
        model output에 대한 input의 gradient 반환

        Args:
            inputs: image which has shape (batch #, H, W, C)
        Returns:
            gradient which has shape (batch #, H, W, 1)
        """
        # self.model.inputs[0]와 inputs 간의 shape 일치 여부 확인도 가능
        input_shape = inputs.shape  # shape (batch #, H, W, C)

        with tf.GradientTape() as tape:
            # out has shape (batch #, 1)
            # final_conv_out has shape (batch #, H', W', C'), where H', W', C' represents H, W, C after convolutions
            out, final_conv_out = self.model(inputs)

        grad = tape.gradient(out, final_conv_out)  # d_out / d_final_conv_out which has shape (batch #, H', W', C')
        # get summation of grad of each feature map which has shape (batch #, C')
        feature_map_grad_sum = tf.math.reduce_sum(tf.math.reduce_sum(grad, axis=1), axis=1)
        # get mean by calculating sum / (H' * W') which has shape (batch #, C')
        # the result equals to weight of feature map
        feature_map_grad_mean = feature_map_grad_sum / (final_conv_out.shape[1] * final_conv_out.shape[2])

        image_shape = (input_shape[1], input_shape[2])  # (H, W)
        grad_cam_shape = (input_shape[0], input_shape[1], input_shape[2])  # (batch #, H, W)
        grad_cam = tf.zeros(grad_cam_shape)  # (batch #, H, W)
        for c in range(final_conv_out.shape[2]):
            weighted_feature_map = feature_map_grad_mean[:, c] * final_conv_out[:, :, :, c]  # shape (batch #, H', W')
            expand_feature_map = tf.expand_dims(weighted_feature_map, axis=-1)  # shape (batch #, H', W', 1)
            resize_feature_map = [tf.image.resize(x, image_shape, method=tf.image.ResizeMethod.BILINEAR)
                                  for x in expand_feature_map]
            grad_cam += tf.squeeze(tf.concat(resize_feature_map, axis=0))

        grad_cam = tf.expand_dims(grad_cam, axis=-1)  # shape (batch #, H, W, 1)
        return grad_cam
