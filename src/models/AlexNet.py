"""AlexNet 구현

논문 제목: ImageNet Classification with Deep Convolutional Neural Networks

* input image size = 227 x 227 (논문 224 x 224는 오타)
* ((input image size - kernel size) / stride length) + 1
ReLU: tanh와 같은 saturating한 활성화 함수보다 ReLU처럼 non-saturating 활성화 함수를 쓰는게
      같은 성능까지의 필요 epochs가 굉장히 적었다 (즉 학습이 빠름)
Memory issue: 중간에 layer를 나누어 각각의 GPU에 할당
Local response normalization: 인접한 kernel 별 결과값의 제곱한 값의 sum으로 kernel 결과값을 나눈다.
                             일부 layer들에서 ReLU 적용 후에 이러한 normalization 적용했음.
                            brightness normalization이라고 표현함
Overlapping pooling: 말 그대로 overlapping 시키는건데 이것이 overfitting 좀 억제한 것으로 생각된다고 함.


Todo:
    * AlexNet 전체 flow 완성 => 첫번째 Conv부터 Separate 되어야하는데 아직 안돼있음
    * SeparateConv2D에서 각각 다른 GPU에 deploy 하는 부분은 구현되어 있지 않음
    * Local response normalization 구현 시 자료형 변환에 의한 성능 저하 가능성 체크
    * 성능 평가
"""
import typing

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.python.framework.ops import EagerTensor


class LocalResponseNormalization(Layer):
    """Brightness normalization

    The normalization equation is from the paper (AlexNet)\n
    k, n, alpha, beta is pre-set according to the paper\n
    현대의 CNN은 batch normalization 기법을 많이 씀
    """
    def __init__(self, k: int = 2, n: int = 5, alpha: float = 1e-4, beta: float = 0.75):
        super(LocalResponseNormalization, self).__init__()
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs: EagerTensor, *args, **kwargs) -> EagerTensor:
        """Normalize the inputs

        Args:
            inputs: (batch size x width x height x channel)
        Returns:
            normalized inputs
        """
        N = tf.shape(inputs)[3]  # number of channels
        out = tf.identity(inputs).numpy()  # copy and type-cast (maybe performance degradation)
        for i in range(N):
            divisor = self.k + self.alpha * \
                      tf.reduce_sum(inputs[:, :, :, tf.maximum(0, i-int(self.n/2)):tf.maximum(N-1, int(i+self.n/2))] ** 2)
            out[:, :, :, i] = inputs[:, :, :, i] / (divisor ** self.beta)

        out = tf.convert_to_tensor(out)
        return out


class SeparateConv2D(Layer):
    """Separate Conv2D output for memory reduction

    Divide input into N sub-inputs (axis = channel) and apply conv2d to each of them.\n
    By deploying each input to different GPU, the memory consumption can be reduced. \n

    Args:
        filters: total numer of filters
        kernel_size: (e.g., (N, N))
        strides:
        n: each out has (filters / 2) filters
    """
    def __init__(self, filters: int, kernel_size: typing.Tuple[int, int], strides: int, padding: str = "valid",  n: int = 2):
        if filters % 2 != 0:
            raise ValueError

        super(SeparateConv2D, self).__init__()
        self.n = n
        self.conv_list = []
        self.padding = padding
        for i in range(n):
            self.conv_list.append(layers.Conv2D(filters/n, kernel_size, strides=strides, padding=padding))

    def call(self, inputs, *args, **kwargs) -> EagerTensor:
        """Separate and apply Conv2D to inputs

        Args:
            inputs: (batch size x width x height x channel)
        Returns:
            sub inputs
        """
        interval = int(tf.shape(inputs)[3] / self.n)
        sub_inputs = []

        for i in range(self.n):
            start = interval * i
            end = interval * (i+1)
            sub_inputs.append(self.conv_list[i](inputs[:, :, :, start:end]))

        return sub_inputs


class AlexNet(Model):
    """
    Response-normalization layers follow the first and second Conv layers.\n
    Max-pooling layers follow response-normalization layers and fifth Conv layer.\n
    ReLU applied to output of every conv and fully connected layer.\n
    layer 별로 미리 정리해두는게 나을듯

    """
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = SeparateConv2D(filters=96, kernel_size=(11, 11), strides=4, n=2)
        self.act1 = layers.ReLU()
        self.norm1 = LocalResponseNormalization()  # normalization after ReLU
        self.pool1 = layers.MaxPool2D((3, 3), strides=2)

        self.conv2_1 = layers.Conv2D(kernel_size=(5, 5), filters=128, padding="same")
        self.conv2_2 = layers.Conv2D(kernel_size=(5, 5), filters=128, padding="same")
        self.concat2 = layers.Concatenate(axis=3)
        self.act2 = layers.ReLU()
        self.norm2 = LocalResponseNormalization()
        self.pool2 = layers.MaxPool2D((3, 3), strides=2)

        self.conv3 = SeparateConv2D(filters=384, kernel_size=(3, 3), strides=1, padding="same")
        self.act3 = layers.ReLU()

        self.conv4_1 = layers.Conv2D(kernel_size=(3, 3), filters=192, padding="same")
        self.conv4_2 = layers.Conv2D(kernel_size=(3, 3), filters=192, padding="same")
        self.act4 = layers.ReLU()

        self.conv5_1 = layers.Conv2D(kernel_size=(3, 3), filters=128, padding="same")
        self.conv5_2 = layers.Conv2D(kernel_size=(3, 3), filters=128, padding="same")
        self.act5 = layers.ReLU()
        self.pool5 = layers.MaxPool2D((3, 3), strides=2)
        self.concat5 = layers.Concatenate(axis=3)
        self.flatten5 = layers.Flatten()

        self.fc6 = layers.Dense(4096)
        self.fc7 = layers.Dense(4096)
        self.fc8 = layers.Dense(1000, activation="softmax")

    def call(self, inputs, training=None, mask=None) -> EagerTensor:
        # First layer
        out = self.conv1(inputs)
        out = self.act1(out)
        out = self.norm1(out)
        out1 = self.pool1(out[0])
        out2 = self.pool1(out[1])

        # Second layer
        out1 = self.conv2_1(out1)
        out2 = self.conv2_2(out2)
        out = self.concat2((out1, out2))
        out = self.act2(out)
        out = self.norm2(out)
        out = self.pool2(out)

        # Third layer
        out = self.conv3(out)
        out = self.act3(out)

        # Fourth layer
        out1 = self.conv4_1(out[0])
        out2 = self.conv4_2(out[1])
        out1 = self.act4(out1)
        out2 = self.act4(out2)

        # Fifth layer
        out1 = self.conv5_1(out1)
        out2 = self.conv5_2(out2)
        out1 = self.act5(out1)
        out2 = self.act5(out2)
        out1 = self.pool5(out1)
        out2 = self.pool5(out2)
        out = self.concat5((out1, out2))
        out = self.flatten5(out)

        # Fully connected layers
        out = self.fc6(out)
        out = self.fc7(out)
        out = self.fc8(out)

        return out

    def get_config(self):
        pass


if __name__ == "__main__":
    data = tf.random.normal((1, 224, 224, 3))
    model = AlexNet()
    out = model(data)
    print(model.summary())
    print(tf.shape(out))
