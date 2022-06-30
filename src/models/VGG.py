"""VGG 구현

논문 제목: Very deep convolutional networks for large-scale image recognition

구성요소:

conv: 오직 3x3 filter로만 이루어진 간단한 모델이지만 deeper하게 구성

image size: 224 x 224


"""
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class VGG(Model):
    def __init__(self, layer1_num: int, layer2_num: int, layer3_num: int, layer4_num: int, layer5_num: int):
        """
        Local response normalization (LRN) 추가는 고려하지 않았으나 VGG16, VGG19 구현엔 문제없음
        Args:
            layer1_num: conv layer # of layer 1
            layer2_num: conv layer # of layer 2
            layer3_num: conv layer # of layer 3
            layer4_num: conv layer # of layer 4
            layer5_num: conv layer # of layer 5
        """
        super().__init__()
        self.sequence = keras.Sequential()

        for _ in range(layer1_num):
            self.sequence.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same",
                                            activation="relu"))
        self.sequence.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"))

        for _ in range(layer2_num):
            self.sequence.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same",
                                            activation="relu"))
        self.sequence.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"))

        for _ in range(layer3_num):
            self.sequence.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same",
                                            activation="relu"))
        self.sequence.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"))

        for _ in range(layer4_num):
            self.sequence.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same",
                                            activation="relu"))
        self.sequence.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"))

        for _ in range(layer5_num):
            self.sequence.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same",
                                            activation="relu"))
        self.sequence.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"))
        self.sequence.add(layers.Flatten())
        self.sequence.add(layers.Dense(4096, activation="relu"))
        self.sequence.add(layers.Dense(4096, activation="relu"))
        self.sequence.add(layers.Dense(1000, activation="softmax"))

    def call(self, inputs, training=None, mask=None):
        return self.sequence(inputs)


def VGG16():
    return VGG(2, 2, 3, 3, 3)  # 2 + 2 + 3 + 3 + 3 + 3 (fc layers) = 16


def VGG19():
    return VGG(2, 2, 4, 4, 4)  # 2 + 2 + 4 + 4 + 4 + 3 (fc layers) = 19


if __name__ == "__main__":
    rand_input = tf.random.normal((1, 224, 224, 3))
    model = VGG19()
    print(model(rand_input).shape)
    print(model.summary())
