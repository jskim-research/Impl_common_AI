import tensorflow as tf
from src.models import Resnet, Resnet_update


def test_resnet():
    """resnet 모델 검증"""
    resnet18 = Resnet.resnet_18()
    resnet34 = Resnet.resnet_34()
    resnet50 = Resnet.resnet_50()
    resnet101 = Resnet.resnet_101()
    resnet152 = Resnet.resnet_152()

    func_resnet18 = Resnet_update.resnet18()
    func_resnet34 = Resnet_update.resnet34()
    func_resnet50 = Resnet_update.resnet50()
    func_resnet101 = Resnet_update.resnet101()
    func_resnet152 = Resnet_update.resnet152()

    batch_num = 8
    tmp_input = tf.random.normal((batch_num, 224, 224, 3))  # (batch #, H, W, C)

    # ImageNet 대상이므로 1000개의 class에 대한 결과가 나와야 함
    assert resnet18(tmp_input).shape == tf.TensorShape([batch_num, 1, 1, 1000])
    assert resnet34(tmp_input).shape == tf.TensorShape([batch_num, 1, 1, 1000])
    assert resnet50(tmp_input).shape == tf.TensorShape([batch_num, 1, 1, 1000])
    assert resnet101(tmp_input).shape == tf.TensorShape([batch_num, 1, 1, 1000])
    assert resnet152(tmp_input).shape == tf.TensorShape([batch_num, 1, 1, 1000])
    assert func_resnet18(tmp_input).shape == tf.TensorShape([batch_num, 1, 1, 1000])
    assert func_resnet34(tmp_input).shape == tf.TensorShape([batch_num, 1, 1, 1000])
    assert func_resnet50(tmp_input).shape == tf.TensorShape([batch_num, 1, 1, 1000])
    assert func_resnet101(tmp_input).shape == tf.TensorShape([batch_num, 1, 1, 1000])
    assert func_resnet152(tmp_input).shape == tf.TensorShape([batch_num, 1, 1, 1000])

