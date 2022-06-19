"""TFRecord 관련 클래스 모음
"""
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from abc import *

import tensorflow.python.data


def image_feature(value):
    """
    image => encoding => bytes 변환

    Args:
        value: image 형태의 데이터 (e.g., shape: width x height x channel)

    Returns:
        bytes
    """
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """
    string/bytes => bytes 변환

    Args:
        value: string or bytes

    Returns:
        bytes
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def int64_feature(value):
    """
    int64 => bytes 변환

    Args:
        value: int64

    Returns:
        bytes
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class DogsCatsRecord:
    """
    Dogs-vs-Cats 데이터를 TFRecord로 변환 및 로드하는 함수 구현
    """
    def __init__(self, file_path: str, save_path: str, num_samples: int):
        self.feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
        self.num_samples = num_samples  # number of samples for each tfrecord
        self.total_num = len(os.listdir(file_path))  # total number of files
        self.file_path = file_path  # files that will be saved as tfrecord
        self.save_path = save_path  # where to save the tfrecord

    def get_data_num(self):
        """tfrecord 대상 데이터 개수 반환"""
        return self.total_num

    def create_example(self, image, label):
        """
        데이터를 bytes로 변환

        Args:
            image:
            label:

        Returns:

        """
        feature = {
            "image": image_feature(image),
            "label": int64_feature(label),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def parse_example(self, example):
        example = tf.io.parse_single_example(example, self.feature_description)
        example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
        return example

    def make_records(self):
        file_list = os.listdir(self.file_path)
        num_tfrecords = len(file_list) // self.num_samples
        if len(file_list) % self.num_samples:
            num_tfrecords += 1  # add one recorde if there are any remaining samples (case: 4096, 4096, ..., 4100)

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        for tfrec_num in range(num_tfrecords):
            samples = file_list[(tfrec_num * self.num_samples):((tfrec_num + 1) * self.num_samples)]
            with tf.io.TFRecordWriter(self.save_path + "file_%.2i-%i.tfrec" % (tfrec_num, len(file_list))) as writer:
                for sample in samples:  # sample = file_name
                    image = tf.io.decode_jpeg(tf.io.read_file(self.file_path + sample))
                    image = tf.image.resize(image, [112, 112])  # method default = bilinear
                    image = tf.cast(image, dtype=tf.uint8)  # resize 후에 float가 되는데 JPEG encode 시에는 uint8로 써야함
                    if sample.split(".")[0] == "cat":
                        label = 0
                    else:
                        label = 1

                    example = self.create_example(image, label)
                    writer.write(example.SerializeToString())

    def normalize(self, example):
        """tfrecord 데이터셋 전처리 및 (x, y) 형태 생성"""
        example['image'] = example['image'] / 255  # 0~1 정규화
        input = example['image']
        label = example.pop('label')
        return input, label

    def load_records(self) -> tf.data.Dataset:
        """
        Returns:
            dataset which consists of records
        """
        file_list = os.listdir(self.save_path)
        tfrecord_path = [self.save_path + fn for fn in file_list]
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = raw_dataset.map(self.parse_example)
        parsed_dataset = parsed_dataset.map(self.normalize)
        return parsed_dataset


if __name__ == "__main__":
    # dogs-vs-cats tfrecord 예제
    file_path_test = "../data/dogs-vs-cats/train/"
    save_path_test = "../data/dogs-vs-cats/tf_records/train/"
    num_samples_test = 4096
    record = DogsCatsRecord(file_path_test, save_path_test, num_samples_test)
    record.make_records()
    dataset = record.load_records()
    dataset = dataset.shuffle(25000).batch(32)
    for x, y in dataset:
        print(x.shape)


