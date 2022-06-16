"""데이터 로딩 함수 구현

ToDo:
    * dogs-vs-cats 로딩함수 구현 필요 (https://www.kaggle.com/competitions/dogs-vs-cats/data)
    * Cifar100 return => class로 대체
"""
import pickle
import util
import numpy as np
import typing
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


def cifar10() -> typing.Tuple[np.ndarray, typing.List[int], typing.List[str]]:
    """

    need to see statistic of the data

    Returns:
        data (50,000 x 3,072), labels (50,000), label_names (50,000)
    """
    data_list = []
    label_list = []
    label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
                   "ship", "truck"]

    for i in range(1, 6):
        filepath = str(util.get_root_folder_path()) + "/data/cifar-10/data_batch_1"
        with open(filepath, "rb") as f:
            cifar10_dict = pickle.load(f, encoding="bytes")
            data_list.append(cifar10_dict[b"data"])
            label_list += cifar10_dict[b"labels"]

    data_list = np.concatenate(data_list)
    return data_list, label_list, label_names


def cifar100() -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray, np.ndarray,
                               typing.List[str], typing.List[str]]:
    """

    500 training images and 100 testing images per class (total 50,000 + 10,000) x 3072 pixels (3 chan, 32x32)

    Returns:
        train_data, train_fine_labels, train_coarse_labels,
        test_data, test_fine_labels, test_coarse_labels,
        fine_label_names, coarse_label_names
    """
    fine_label_names = ["beaver", "dolphin", "otter", "seal", "whale",
                        "aquarium fish", "flatfish", "ray", "shark", "trout",
                        "orchids", "poppies", "roses", "sunflowers", "tulips",
                        "bottles", "bowls", "cans", "cups", "plates",
                        "apples", "mushrooms", "oranges", "pears", "sweet peppers",
                        "clock", "computer keyboard", "lamp", "telephone", "television",
                        "bed", "chair", "couch", "table", "wardrobe",
                        "bee", "beetle", "butterfly", "caterpillar", "cockroach",
                        "bear", "leopard", "lion", "tiger", "wolf",
                        "bridge", "castle", "house", "road", "skyscraper",
                        "cloud", "forest", "mountain", "plain", "sea",
                        "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
                        "fox", "porcupine", "possum", "racoon", "skunk",
                        "crab", "lobster", "snail", "spider", "worm",
                        "baby", "boy", "girl", "man", "woman",
                        "crocodile", "dinosaur", "lizard", "snake", "turtle",
                        "hamster", "mouse", "rabbit", "shrew", "squirrel",
                        "maple", "oak", "palm", "pine", "willow",
                        "bicycle", "bus", "motorcycle", "pickup truck", "train",
                        "lawn-mower", "rocket", "streetcar", "tank", "tractor"]
    coarse_label_names = ["aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
                          "household electrical devices", "household furniture", "insects", "large carnivores",
                          "large man-made outdoor things", "large natural outdoor scenes",
                          "large omnivores and herbivores",
                          "medium-sized mammals", "non-insect invertebrates", "people", "reptiles",
                          "small mammals", "trees", "vehicles 1", "vehicles 2"]

    train_filepath = str(util.get_root_folder_path()) + "/data/cifar-100/train"
    test_filepath = str(util.get_root_folder_path()) + "/data/cifar-100/test"
    with open(train_filepath, "rb") as f:
        cifar100_dict = pickle.load(f, encoding="bytes")
        train_data = cifar100_dict[b"data"]
        train_fine_labels = np.array(cifar100_dict[b"fine_labels"])
        train_coarse_labels = np.array(cifar100_dict[b"coarse_labels"])

    with open(test_filepath, "rb") as f:
        cifar100_dict = pickle.load(f, encoding="bytes")
        test_data = cifar100_dict[b"data"]
        test_fine_labels = np.array(cifar100_dict[b"fine_labels"])
        test_coarse_labels = np.array(cifar100_dict[b"coarse_labels"])

    return (train_data, train_fine_labels, train_coarse_labels,
            test_data, test_fine_labels, test_coarse_labels,
            fine_label_names, coarse_label_names)


def image_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])  
        # encode를 해서 저장하므로 메모리를 좀 줄일 수 있을 듯
    )


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode("utf-8")]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_example(image, path, file_name):
    feature = {
        # "image": image_feature(image),
        "path": bytes_feature(path),
        "filename": bytes_feature(file_name),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        # "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "filename": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    # example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    return example


def Make_TFRecord(file_path: str, save_path: str, num_samples: int) -> None:
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
        samples = file_list[(tfrec_num * num_samples):((tfrec_num+1) * num_samples)]
        with tf.io.TFRecordWriter(save_path + "file_%.2i-%i.tfrec" % (tfrec_num, len(file_list))) as writer:
            for sample in samples:  # sample = file_name
                image = tf.io.decode_jpeg(tf.io.read_file(file_path + sample))
                example = create_example(image, file_path, sample)
                writer.write(example.SerializeToString())

    raw_dataset = tf.data.TFRecordDataset(f"{file_path}/file_00-{num_samples}")
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

    for features in parsed_dataset.take(1):
        print(features)
        for key in features.keys():
            if key != "image":
                print(f"{key}: {features[key]}")

        # print(f"Image shape: {features['image'].shape}")
        # plt.figure(figsize=(7, 7))
        # plt.imshow(features["image"].numpy())
        # plt.show()


def dogs_cats(training: bool) -> typing.Tuple[tf.data.Dataset, int]:
    """
    dogs-vs-cats 데이터 로드

    Args:
        training: if True, load training data. Else, load test data.
    Returns:
        tensorflow dataset, number of data
    """
    if training:
        path = "../data/dogs-vs-cats/train/"
    else:
        path = "../data/dogs-vs-cats/test1/"

    file_list = os.listdir(path)

    def dogs_cats_generator():
        """
        Returns:
            generator

        """
        for idx, fn in enumerate(file_list):
            img = cv2.imread(path + fn, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (112, 112))
            img = tf.convert_to_tensor(img, dtype=tf.float32) / 255  # typecast and normalization
            label = 0
            if fn.split(".")[0] == "cat":
                label = 0
            else:
                label = 1
            yield img, label

    dataset = tf.data.Dataset.from_generator(dogs_cats_generator, (tf.float32, tf.int16),
                                             output_shapes=([112, 112, 3], []))

    return dataset, len(file_list)


if __name__ == "__main__":
    train_img, *rest = dogs_cats()
    print(train_img.shape)

