# coding=utf-8

import pickle
import numpy as np
import os
import cv2


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        x = dict[b'data']
        y = dict[b'labels']
        x = np.reshape(x, (10000, 3, 32, 32))
        x = np.transpose(x, (0, 2, 3, 1))
        y = np.array(y)
    return x, y


def save(dir, x, y):
    assert x.shape[0] == y.shape[0], 'x num is not equal with y'
    for i, label in enumerate(y):
        sub_dir = os.path.join(dir, str(label))
        if os.path.exists(sub_dir):
            pass
        else:
            os.makedirs(sub_dir)
        path = os.path.join(sub_dir, str(i) + '.png')
        image = cv2.cvtColor(x[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)
        if (i + 1) % 100 == 0:
            print(i)


def parse(input_dir, output_dir, file_names):
    for name in file_names:
        file = os.path.join(input_dir, name)
        x, y = unpickle(file)
        save(output_dir, x, y)


if __name__ == '__main__':
    train = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test = ['test_batch']
    input_dir = './cifar/original'
    train_dir = './cifar/train'
    test_dir = './cifar/test'

    parse(input_dir, test_dir, test)
