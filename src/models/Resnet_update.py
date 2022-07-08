"""Resnet 구현

이전에 구현한 resnet 갱신 (tensorflow API 형태)

ToDo:
    * layer 생성 시 name 설정
"""
import keras
import tensorflow as tf
import typing
from tensorflow.keras import layers


def identity_block2d(input_tensor: tf.Tensor, filters: int) -> tf.Tensor:
    """
    2개의 Conv. 와 Residual connection로 이루어진 block.
    자기 자신을 residual connection으로 다시 연결하기 때문에 identity mapping이라 불림

    Args:
        input_tensor: tensor which has shape (batch #, H, W, C)
        filters: filters of two Conv.

    Returns:
    """

    out = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=1)(input_tensor)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)

    # output + shortcut 에서 activation 계산
    out = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=1)(out)
    out = layers.BatchNormalization()(out)

    if input_tensor.shape[3] != filters:  # residual connection channel # 불일치 시
        # 1x1 Conv. 로 channel # 맞춤
        input_tensor = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=1)(input_tensor)
        input_tensor = layers.BatchNormalization()(input_tensor)
    out = out + input_tensor
    out = layers.ReLU()(out)
    return out


def bottleneck_block2d(input_tensor: tf.Tensor, filters: int) -> tf.Tensor:
    """
    1x1 Conv. (channel 수 감소) => 3x3 Conv. (channel 수 유지) => 1x1 Conv. (channel 수 원상복구)
    3x3 Conv. 하기 전 연산 등을 줄이기 위해 1x1 Conv. 로 channel 수를 줄인다. (bottleneck 이름의 의미)
    Args:
        input_tensor: tensor which has shape (batch #, H, W, C)
        filters: reduced filters

    Returns:
    """
    out = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(input_tensor)
    out = layers.BatchNormalization()(out)

    out = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(out)
    out = layers.BatchNormalization()(out)

    out = layers.Conv2D(filters=filters*4, kernel_size=(1, 1), strides=1, padding="same")(out)
    out = layers.BatchNormalization()(out)

    # output + shortcut 에서 activation 계산
    if input_tensor.shape[3] != filters*4:  # residual connection channel # 불일치 시
        input_tensor = layers.Conv2D(filters=filters*4, kernel_size=(1, 1), strides=1, padding="same")(input_tensor)
        input_tensor = layers.BatchNormalization()(input_tensor)

    out = out + input_tensor
    out = layers.ReLU()(out)
    return out


def resnet(block: typing.Callable, filters: typing.List[int], layer_nums: typing.List[int]):
    """

    Args:
        block: identity_block2d or bottleneck_block2d
        filters: filter # of each layer of resnet
        layer_nums: layer #

    Returns:
    """
    # if block isinstance of identity_block2d? bottleneck_block2d? (예외 case)
    # len(filters) == len(layer_nums)?
    
    input_tensor = layers.Input(shape=(224, 224, 3))

    # 논문에서 처음에 64 filter로 시작함
    out = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same")(input_tensor)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(out)

    # for문, if문으로 감싸니 모델링에 대한 가독성이 그리 좋지 않다.
    for layer_idx, (_filter, _layer_num) in enumerate(zip(filters, layer_nums)):
        for block_idx in range(_layer_num):
            out = block(out, _filter)
        if layer_idx == len(layer_nums)-1:
            out = layers.AvgPool2D(pool_size=(7, 7))(out)
        else:
            out = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(out)

    out = layers.Dense(1000, activation="softmax")(out)
    return keras.Model(inputs=[input_tensor], outputs=[out])


def resnet18():
    return resnet(identity_block2d, filters=[64, 128, 256, 512], layer_nums=[2, 2, 2, 2])


def resnet34():
    return resnet(identity_block2d, filters=[64, 128, 256, 512], layer_nums=[3, 4, 6, 3])


def resnet50():
    return resnet(bottleneck_block2d, filters=[64, 128, 256, 512], layer_nums=[3, 4, 6, 3])


def resnet101():
    return resnet(bottleneck_block2d, filters=[64, 128, 256, 512], layer_nums=[3, 4, 23, 3])


def resnet152():
    return resnet(bottleneck_block2d, filters=[64, 128, 256, 512], layer_nums=[3, 8, 36, 3])


if __name__ == "__main__":
    input_T = tf.random.normal((1, 224, 224, 3))
    model = resnet152()
    print(model.summary())
    print(model(input_T).shape)
