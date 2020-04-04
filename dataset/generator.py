# coding=utf-8

from functools import partial
import numpy as np
import random
import cv2
import traceback
import os


class Generator(object):
    def __init__(self, num_classes, is_shuffle, is_horizontal_flip, is_random_crop, is_random_cutout):
        self.num_classes = num_classes
        self.is_shuffle = is_shuffle
        self.is_horizontal_flip = is_horizontal_flip
        self.is_random_crop = is_random_crop
        self.is_random_cutout = is_random_cutout

    def random_horizontal_flip(self, image):
        if random.random() <= 0.5:
            return cv2.flip(image, 1)
        else:
            return image

    def random_crop(self, image, padding=4):
        if random.random() <= 0.5:
            image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            x = np.random.randint(0, padding * 2)
            y = np.random.randint(0, padding * 2)
            image = image[y:y + 32, x:x + 32]
            return image
        else:
            return image

    def random_cutout(self, image, offset=8):
        if random.random() <= 0.5:
            h, w, _ = image.shape
            x = np.random.randint(0, w - offset)
            y = np.random.randint(0, h - offset)
            image[y:y + offset, x:x + offset, :] = 0
            return image
        else:
            return image

    def preprocess(self, x, y):
        x = x.astype('float32') / 255
        x -= (0.4914, 0.4822, 0.4465)
        x /= (0.2023, 0.1994, 0.2010)
        return x, y

    def load(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def sample(self, image_path, label):
        image = self.load(image_path)
        image = np.array(image, dtype=np.float32)
        one_hot = np.zeros(self.num_classes, dtype=np.int32)
        one_hot[label] = 1
        label = one_hot
        image, label = self.preprocess(image, label)

        if self.is_horizontal_flip:
            image = self.random_horizontal_flip(image)

        if self.is_random_crop:
            image = self.random_crop(image)

        if self.is_random_cutout:
            image = self.random_cutout(image)

        return image, label

    def generate(self, dataset):
        if self.is_shuffle:
            random.shuffle(dataset)
        for image_path, label in dataset:
            try:
                data = self.sample(image_path, label)
                yield data
            except Exception as e:
                traceback.print_tb(e.__traceback__)


def get_dataset(dir):
    dataset = list()
    for label in os.listdir(dir):
        sub_dir = os.path.join(dir, label)
        for name in os.listdir(sub_dir):
            path = os.path.join(sub_dir, name)
            dataset.append((path, int(label)))
    return dataset


def generator(dir, num_classes, is_shuffle=False, is_horizontal_flip=False, is_random_crop=False, is_random_cutout=False):
    dataset = get_dataset(dir)
    gt = Generator(num_classes=num_classes,
                   is_shuffle=is_shuffle,
                   is_horizontal_flip=is_horizontal_flip,
                   is_random_crop=is_random_crop,
                   is_random_cutout=is_random_cutout)
    return partial(gt.generate, dataset=dataset)
