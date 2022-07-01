"""Resnet 구현

논문 제목: Deep Residual Learning for Image Recognition

배경: 단순히 DL model의 layer를 깊게 할 경우 gradient vanishing 문제로 성능이 감소한다.
그러나 identity mapping을 통해 layer를 깊게 할 경우
identity mapping에 의한 성능 저하가 없을 뿐 아니라 추가적인 경사하강법에 의한 성능 향상을 노릴 수 있다.
이에 따라 훈련이 잘 됨.

"""
import typing
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class IdentityBlock(Layer):
    def __init__(self, channel: int, shortcut_projection: bool):
        """
        Args:
            channel: number of channels
            shortcut_projection: if the output shape does not match with shortcut shape, projection needed.
        """
        super().__init__()
        self.conv1 = layers.Conv2D(filters=channel, kernel_size=(3, 3), strides=1, padding="same", activation="relu")
        # output + shortcut 에서 activation 계산
        self.conv2 = layers.Conv2D(filters=channel, kernel_size=(3, 3), strides=1, padding="same")
        self.shortcut_projection = shortcut_projection
        if self.shortcut_projection:
            # matching the shape using projection conv. (zero padding도 쓸 수 있으나 이 방법이 좀 더 성능 향상)
            self.projection = layers.Conv2D(filters=channel, kernel_size=(1, 1), strides=1, padding="same")
        self.norm = layers.ReLU()

    def call(self, inputs, *args, **kwargs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        if self.shortcut_projection:
            inputs = self.projection(inputs)
        return self.norm(out + inputs)


class BottleneckBlock(Layer):
    """
    이전 layer의 channel을 1x1 conv가 줄였다가 다시 1x1 conv가 늘리기 때문에 처음에 정보가 막힌다. (아마도 bottleneck 의미)
    이렇게 한 이유는 중간의 3x3 conv가 적은 dimension의 data에 대해 작업하도록 하기 위해서다.
    """
    def __init__(self, channel: int, shortcut_projection: bool):
        """

        Args:
            channel: number of channels for first conv. number of output channel is 4 * channel.
            shortcut_projection: if the output shape does not match with shortcut shape, projection needed.
        """
        super().__init__()
        self.conv1 = layers.Conv2D(filters=channel, kernel_size=(1, 1), strides=1, padding="same", activation="relu")
        self.conv2 = layers.Conv2D(filters=channel, kernel_size=(3, 3), strides=1, padding="same", activation="relu")
        # output + shortcut 에서 activation 계산
        self.conv3 = layers.Conv2D(filters=channel*4, kernel_size=(1, 1), strides=1, padding="same")
        self.shortcut_projection = shortcut_projection
        if self.shortcut_projection:
            self.projection = layers.Conv2D(filters=channel*4, kernel_size=(1, 1), strides=1, padding="same")
        self.norm = layers.ReLU()

    def call(self, inputs, *args, **kwargs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.shortcut_projection:
            inputs = self.projection(inputs)
        return self.norm(out + inputs)


class ResNet(Model):
    """
    논문에 명시된 대로 shortcut projection이 필요한 경우는 각 conv layer를 넘어갈 때만이라고 가정함.
    """
    def __init__(self, identity_mapping: typing.Type[Layer],
                 conv2_chan: int, conv2_mapping: int,
                 conv3_chan: int, conv3_mapping: int,
                 conv4_chan: int, conv4_mapping: int,
                 conv5_chan: int, conv5_mapping: int):
        """

        Args:
            identity_mapping: This should be either IdentityBlock or BottleneckBlock. Resnet consists of these blocks.
            conv2_chan: number of channels at layer 2
            conv2_mapping: number of mapping blocks at layer 2
            conv3_chan: number of channels at layer 3
            conv3_mapping: number of mapping blocks at layer 3
            conv4_chan: number of channels at layer 4
            conv4_mapping: number of mapping blocks at layer 4
            conv5_chan: number of channels at layer 5
            conv5_mapping: number of mapping blocks at layer 5
        """
        super().__init__()
        self.sequence = keras.Sequential()
        first_filters = 64
        self.sequence.add(layers.Conv2D(filters=first_filters, kernel_size=(7, 7), strides=2, padding="same", activation="relu"))

        self.sequence.add(layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"))
        for idx in range(conv2_mapping):
            # 초반에는 Block에 따라 short cut projection을 하느냐 마느냐가 결정된다.
            if type(identity_mapping) == IdentityBlock:
                self.sequence.add(identity_mapping(channel=conv2_chan, shortcut_projection=(conv2_chan != first_filters)))
            else:
                self.sequence.add(identity_mapping(channel=conv2_chan, shortcut_projection=(conv2_chan*4 != first_filters)))

        self.sequence.add(layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"))
        for idx in range(conv3_mapping):
            # 첫 번째 block만 shortcut_projection 활성화
            self.sequence.add(identity_mapping(channel=conv3_chan, shortcut_projection=(idx == 0)))

        self.sequence.add(layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"))
        for idx in range(conv4_mapping):
            # 첫 번째 block만 shortcut_projection 활성화
            self.sequence.add(identity_mapping(channel=conv4_chan, shortcut_projection=(idx == 0)))

        self.sequence.add(layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"))
        for idx in range(conv5_mapping):
            # 첫 번째 block만 shortcut_projection 활성화
            self.sequence.add(identity_mapping(channel=conv5_chan, shortcut_projection=(idx == 0)))

        self.sequence.add(layers.AvgPool2D(pool_size=(7, 7)))
        self.sequence.add(layers.Dense(1000, activation="softmax"))

    def call(self, inputs, training=None, mask=None):
        return self.sequence(inputs)


def resnet_18():
    return ResNet(IdentityBlock, 64, 2, 128, 2, 256, 2, 512, 2)


def resnet_34():
    return ResNet(IdentityBlock, 64, 3, 128, 4, 256, 6, 512, 3)


def resnet_50():
    return ResNet(BottleneckBlock, 64, 3, 128, 4, 256, 6, 512, 3)


def resnet_101():
    return ResNet(BottleneckBlock, 64, 3, 128, 4, 256, 23, 512, 3)


def resnet_152():
    return ResNet(BottleneckBlock, 64, 3, 128, 8, 256, 36, 512, 3)


if __name__ == "__main__":
    rand_input = tf.random.normal((1, 224, 224, 3))
    model = resnet_152()
    print(model(rand_input).shape)

