"""GoogLeNet 구현

논문 제목: Going deeper with convolutions

요약:
Inception module
=> 1x1, 3x3, 5x5 로 이루어진 모듈이며 dimension reduction을 위해 1x1 conv를 추가함
=> 장점: 다음 layer에서 다양한 scale의 visual information을 볼 수 있음. 그러면서도 complexity를 너무 증가시키지 않음.

input: 224x224
activation: 모든 layer에서 relu를 사용한 것으로 확인됨

"""
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.models import Model


class InceptionModule(Layer):
    def __init__(self, filter_11: int, reduce_33: int, filter_33: int, reduce_55: int, filter_55: int, filter_pool: int):
        """
        Args:
            filter_11: 1x1 conv filter 개수
            reduce_33: 3x3 conv 이전에 적용할 1x1 conv filter 개수
            filter_33: 3x3 conv filter 개수
            reduce_55: 5x5 conv 이전에 적용할 1x1 conv filter 개수
            filter_55: 5x5 conv filter 개수
            filter_pool: max pooling 이후 1x1 conv filter 개수
        """
        super().__init__()
        self.conv11 = layers.Conv2D(filters=filter_11, kernel_size=(1, 1), padding="same", activation="relu")
        self.conv11_33 = layers.Conv2D(filters=reduce_33, kernel_size=(1, 1), padding="same", activation="relu")  # reduce_33
        self.conv11_55 = layers.Conv2D(filters=reduce_55, kernel_size=(1, 1), padding="same", activation="relu")  # reduce_55
        self.conv33 = layers.Conv2D(filters=filter_33, kernel_size=(3, 3), padding="same", activation="relu")
        self.conv55 = layers.Conv2D(filters=filter_55, kernel_size=(5, 5), padding="same", activation="relu")
        self.pool = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same")
        self.pool_conv11 = layers.Conv2D(filters=filter_pool, kernel_size=(1, 1), padding="same", activation="relu")

    def call(self, inputs, *args, **kwargs):
        input_scale1 = self.conv11(inputs)
        input_scale2 = self.conv33(self.conv11_33(inputs))
        input_scale3 = self.conv55(self.conv11_55(inputs))
        input_scale4 = self.pool_conv11(self.pool(inputs))
        return tf.concat([input_scale1, input_scale2, input_scale3, input_scale4], axis=3)


class GoogLeNet(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same", activation="relu")
        self.pool1 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")
        self.conv2 = layers.Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding="same", activation="relu")
        self.pool2 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")
        self.inception3a = InceptionModule(64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(128, 128, 192, 32, 96, 64)
        self.pool3 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")
        self.inception4a = InceptionModule(192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(256, 160, 320, 32, 128, 128)
        self.pool4 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")
        self.inception5a = InceptionModule(256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(384, 192, 384, 48, 128, 128)
        self.pool5 = layers.AvgPool2D(pool_size=(7, 7))
        self.norm5 = layers.Dropout(0.4)
        self.fc5 = layers.Dense(1000, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.pool3(out)
        out = self.inception4a(out)
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        out = self.inception4e(out)
        out = self.pool4(out)
        out = self.inception5a(out)
        out = self.inception5b(out)
        out = self.pool5(out)
        out = self.norm5(out)
        out = self.fc5(out)
        return out


if __name__ == "__main__":
    model = GoogLeNet()
    inputs = tf.random.normal((1, 224, 224, 3))
    print(model(inputs).shape)
