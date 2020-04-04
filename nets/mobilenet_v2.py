# coding=utf-8

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class Block(layers.Layer):
    def __init__(self, in_planes, out_planes, expansion, stride, weight_decay=1e-4):
        super(Block, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        planes = in_planes * expansion
        self.conv1 = layers.Conv2D(filters=planes,
                                   kernel_size=1,
                                   strides=1,
                                   padding='valid',
                                   use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.DepthwiseConv2D(kernel_size=3,
                                            strides=stride,
                                            padding='same',
                                            use_bias=False,
                                            depthwise_initializer='he_normal',
                                            depthwise_regularizer=keras.regularizers.l2(weight_decay))
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(filters=out_planes,
                                   kernel_size=1,
                                   strides=1,
                                   padding='valid',
                                   use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.bn3 = layers.BatchNormalization()

        if stride == 1 and in_planes != out_planes:
            self.shortcut = keras.Sequential()
            self.shortcut.add(layers.Conv2D(filters=out_planes,
                                            kernel_size=1,
                                            strides=1,
                                            padding='valid',
                                            use_bias=False,
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=keras.regularizers.l2(weight_decay))
                              )
            self.shortcut.add(layers.BatchNormalization())

    def call(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv3(outputs)
        outputs = self.bn3(outputs)
        if self.stride == 1:
            if self.in_planes == self.out_planes:
                outputs += inputs
            else:
                outputs += self.shortcut(inputs)
        return outputs


class MobileNet(keras.Model):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, weight_decay=1e-4):
        super(MobileNet, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = layers.Conv2D(filters=32,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.bn1 = layers.BatchNormalization()
        self.sequential = self._make_layers(in_planes=32)
        self.conv2 = layers.Conv2D(filters=1280,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.bn2 = layers.BatchNormalization()
        self.pooling = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_classes,
                                  activation='softmax',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=keras.regularizers.l2(weight_decay))

    def _make_layers(self, in_planes):
        sequential = keras.Sequential()
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                sequential.add(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return sequential

    def call(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = self.sequential(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = self.pooling(outputs)
        outputs = self.flatten(outputs)
        outputs = self.dense(outputs)
        return outputs


def MobileNetV2(input_shape, num_classes=10, weight_decay=1e-4):
    inputs = keras.Input(shape=input_shape)
    outputs = MobileNet(num_classes=num_classes, weight_decay=weight_decay)(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)
