"""
Cifar10에 대한 성능 테스트를 위해 일부 모델 수정 후 학습 시켜봄
대부분의 모델이 ImageNet (224x224) 이미지 기반이라 이러한 작업 필요
"""

import datasets
import callbacks
import numpy as np
import pandas as pd
from models import GoogLeNet
from tensorflow.keras import layers, losses, metrics, optimizers
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


class SmallGoogLeNet(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="same", activation="relu")
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=1, padding="same")
        self.conv2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same", activation="relu")
        self.pool2 = layers.AvgPool2D()

        # self.conv1 = GoogLeNet.InceptionModule(64, 48, 64, 16, 32, 32)
        # self.pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=1)
        self.flat = layers.Flatten()
        self.fc1 = layers.Dense(10, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.pool1(out)
        out = self.flat(out)
        out = self.fc1(out)
        return out


if __name__ == "__main__":
    model = SmallGoogLeNet()
    data, label, label_name = datasets.cifar10()
    # tensorflow 에선 (높이, 너비, 채널)을 요구
    data = data.reshape((data.shape[0], 3, 32, 32)).transpose(0, 3, 2, 1) / 255.0  # (50,000, 32, 32, 3)
    label = np.array(label)

    X_train, X_test, y_train, y_test = train_test_split(data, label, stratify=label, test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=0.3)

    opt = optimizers.Adam(learning_rate=1e-3)
    loss_func = losses.SparseCategoricalCrossentropy()
    metric_func = metrics.SparseCategoricalAccuracy()

    model.compile(optimizer=opt, loss=loss_func, metrics=[metric_func])
    model.fit(X_train, y_train, validation_data=[X_valid, y_valid], batch_size=8, epochs=10,
              callbacks=[callbacks.GetTensorboardCallback("../tensor_board/cifar10_base_avgpool")])
    print(model.evaluate(X_test, y_test))
    print(model.summary())
