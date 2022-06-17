"""TFRecord 관련 클래스 모음
"""
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from abc import *


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
    def __init__(self):
        self.feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "path": tf.io.FixedLenFeature([], tf.string),
            "filename": tf.io.FixedLenFeature([], tf.string),
        }

    def create_example(self, image, path, file_name):
        feature = {
            "image": image_feature(image),
            "path": bytes_feature(path),
            "filename": bytes_feature(file_name),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def parse_example(self, example):
        example = tf.io.parse_single_example(example, self.feature_description)
        example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
        return example

    def make_records(self, file_path: str, save_path: str, num_samples: int):
        """Make TFRecord of dogs-cats dataset

        Args:
            file_path: where to load
            save_path: where to save
            num_samples: number of samples in each TFRecord file
        Returns:
            return nothing
        """

        file_list = os.listdir(file_path)
        num_tfrecords = len(file_list) // num_samples
        if len(file_list) % num_samples:
            num_tfrecords += 1  # add one recorde if there are any remaining samples

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for tfrec_num in range(num_tfrecords):
            samples = file_list[(tfrec_num * num_samples):((tfrec_num + 1) * num_samples)]
            with tf.io.TFRecordWriter(save_path + "file_%.2i-%i.tfrec" % (tfrec_num, len(file_list))) as writer:
                for sample in samples:  # sample = file_name
                    image = tf.io.decode_jpeg(tf.io.read_file(file_path + sample))
                    example = self.create_example(image, file_path, sample)
                    writer.write(example.SerializeToString())

    def load_records(self, filepath: str):
        """

        Args:
            filepath: where to load records
        """
        raw_dataset = tf.data.TFRecordDataset("../data/dogs-vs-cats/tf_records/train/file_00-25000.tfrec")
        parsed_dataset = raw_dataset.map(self.parse_example)

        print(parsed_dataset.take(1))
        for features in parsed_dataset.take(1):
            print(features)
            for key in features.keys():
                if key != "image":
                    print(f"{key}: {features[key]}")

            print(f"Image shape: {features['image'].shape}")
            plt.figure(figsize=(7, 7))
            plt.imshow(features["image"].numpy())
            plt.show()


if __name__ == "__main__":
    pass

