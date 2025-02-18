"""Densenet 구현

논문 제목: Densely Connected Convolutional Networks

참고사항
- Conv 결과들을 concat 시켜서 정보들을 보존하자
- Bottleneck layer => BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3), each 1x1 conv produce 4k feature-maps (growth_rate=4?)
- Transition layer => layers between blocks => BN-Conv(1x1)-AverPool(2x2)
- Densenet-C: transition layer를 지날 때 feature map 개수 절반 될 때. (theta = 0.5, theta * num feature maps)
- Densenet-BC: bottleneck과 transition layer 모두 theta < 1 일 때
- kernel_initializer는?
"""
import tensorflow as tf
import typing
import keras
from tensorflow.keras import layers


def bottleneck_layer(input_tensor: tf.Tensor, k: int):
    """
    Args:
        input_tensor:
        k: channel's default setting (called growth_rate in the paper)
    Returns:

    """
    out = layers.Conv2D(filters=k*4, kernel_size=(1, 1), strides=1, padding="same")(input_tensor)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.Conv2D(filters=k, kernel_size=(3, 3), strides=1, padding="same")(out)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    return out


def dense_block(input_tensor: tf.Tensor, k: int, bottleneck_num: int):
    """

    Args:
        input_tensor:
        k: channel's default setting (called growth_rate in the paper)
        bottleneck_num: bottleneck layers #
    Returns:
    """
    out = input_tensor
    for idx in range(bottleneck_num):
        out = layers.Concatenate(axis=3)([out, bottleneck_layer(out, k)])
    return out


def transition_layer(input_tensor: tf.Tensor, k: int, theta: float):
    """

    Args:
        input_tensor:
        k: channel's default setting (called growth_rate in the paper)
        theta: (input channel # / output channel #)

    Returns:

    """
    out_filter = int(k * theta)
    out = layers.Conv2D(filters=out_filter, kernel_size=(1, 1), strides=1, padding="same")(input_tensor)
    out = layers.AvgPool2D(pool_size=(2, 2), strides=2, padding="same")(out)
    return out


def densenet(k: int, bottleneck_nums: typing.List[int]):
    """

    Args:
        k: channel's default setting (called growth_rate in the paper)
        bottleneck_nums: list of bottleneck # in dense blocks (len(dense blocks) == len(bottleneck_nums))
    Returns:

    """
    input_tensor = layers.Input(shape=(224, 224, 3))
    out = layers.Conv2D(filters=k, kernel_size=(7, 7), strides=2, padding="same")(input_tensor)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(out)

    for idx in range(len(bottleneck_nums)-1):
        out = dense_block(out, k, bottleneck_nums[idx])
        out = transition_layer(out, k=k, theta=0.5)

    out = layers.AvgPool2D(pool_size=(7, 7))(out)
    out = layers.Dense(1000, activation="softmax")(out)
    return keras.Model(inputs=[input_tensor], outputs=[out])


def densenet121():
    return densenet(32, [6, 12, 24, 16])


def densenet169():
    return densenet(32, [6, 12, 32, 32])


def densenet201():
    return densenet(32, [6, 12, 48, 32])


def densenet264():
    return densenet(32, [6, 12, 64, 48])


if __name__ == "__main__":
    custom_input = tf.random.normal((1, 224, 224, 3))
    model = densenet121()
    print(model.summary())
    print(model(custom_input).shape)
