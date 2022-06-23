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
import tf_record
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


def dogs_cats(init: bool, training: bool) -> typing.Tuple[tf.data.Dataset, int]:
    """
    dogs-vs-cats 데이터 로드

    Args:
        init: if True, make tfrecords first
        training: if True, load training data. Else, load test data.
    Returns:
        tensorflow dataset, number of data
    """
    if training:
        file_path = "../data/dogs-vs-cats/train/"
        save_path = "../data/dogs-vs-cats/tf_records/train/"
    else:
        file_path = "../data/dogs-vs-cats/test1/"
        save_path = "../data/dogs-vs-cats/tf_records/test1/"

    num_samples = 4096
    record = tf_record.DogsCatsRecord(file_path, save_path, num_samples)
    if init:
        record.make_records()
    dataset = record.load_records()
    return dataset, record.get_data_num()


if __name__ == "__main__":
    pass

