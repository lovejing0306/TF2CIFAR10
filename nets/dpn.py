# coding=utf-8

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class GroupConv2D(layers.Conv2D):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 num_group=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.num_group = num_group
        if self.filters % self.num_group != 0:
            raise ValueError("filters must divided by num_group with no remainders!")
        self.input_spec = layers.InputSpec(ndim=4)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        if input_dim % self.num_group != 0:
            raise ValueError("The channel dimension of input tensor must divided by num_group with no remainders!")

        kernel_shape = self.kernel_size + (input_dim // self.num_group, self.filters)
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = layers.InputSpec(ndim=self.rank + 2,
                                           axes={channel_axis: input_dim})
        self.built = True
        self.channel_num = input_dim

    def call(self, inputs):
        filter_in_group = self.filters // self.num_group
        if self.data_format == 'channels_first':
            channel_axis = 1
            input_in_group = self.channel_num // self.num_group
            outputs_list = []
            for i in range(self.num_group):
                outputs = keras.backend.conv2d(inputs[:, i * input_in_group:(i + 1) * input_in_group, :, :],
                                               self.kernel[:, :, :, i * filter_in_group:(i + 1) * filter_in_group],
                                               strides=self.strides,
                                               padding=self.padding,
                                               data_format=self.data_format,
                                               dilation_rate=self.dilation_rate)

                if self.use_bias:
                    outputs = keras.backend.bias_add(outputs,
                                                     self.bias[i * filter_in_group:(i + 1) * filter_in_group],
                                                     data_format=self.data_format)
                outputs_list.append(outputs)

        elif self.data_format == 'channels_last':
            outputs_list = []
            channel_axis = -1
            input_in_group = self.channel_num // self.num_group
            for i in range(self.num_group):
                outputs = keras.backend.conv2d(inputs[:, :, :, i * input_in_group:(i + 1) * input_in_group],
                                               self.kernel[:, :, :, i * filter_in_group:(i + 1) * filter_in_group],
                                               strides=self.strides,
                                               padding=self.padding,
                                               data_format=self.data_format,
                                               dilation_rate=self.dilation_rate)

                if self.use_bias:
                    outputs = keras.backend.bias_add(outputs,
                                                     self.bias[i * filter_in_group:(i + 1) * filter_in_group],
                                                     data_format=self.data_format)
                outputs_list.append(outputs)

        outputs = keras.backend.concatenate(outputs_list, axis=channel_axis)
        return outputs

    def get_config(self):
        config = super(layers.Conv2D, self).get_config()
        config.pop('rank')
        config["num_group"] = self.num_group
        return config


class Bottleneck(layers.Layer):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer, weight_decay=1e-4):
        super(Bottleneck, self).__init__()
        self.last_planes = last_planes
        self.out_planes = out_planes
        self.dense_depth = dense_depth
        self.first_layer = first_layer
        self.conv1 = layers.Conv2D(filters=in_planes,
                                   kernel_size=1,
                                   strides=1,
                                   padding='valid',
                                   use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.bn1 = layers.BatchNormalization()
        self.conv2 = GroupConv2D(filters=in_planes,
                                 kernel_size=3,
                                 strides=stride,
                                 padding='same',
                                 num_group=32,
                                 use_bias=False,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(filters=out_planes + dense_depth,
                                   kernel_size=1,
                                   strides=1,
                                   padding='valid',
                                   use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.bn3 = layers.BatchNormalization()

        if first_layer:
            self.short_cut = keras.Sequential()
            self.short_cut.add(layers.Conv2D(filters=out_planes + dense_depth,
                                             kernel_size=1,
                                             strides=stride,
                                             padding='valid',
                                             use_bias=False,
                                             kernel_initializer='he_normal',
                                             kernel_regularizer=keras.regularizers.l2(weight_decay))
                               )
            self.short_cut.add(layers.BatchNormalization())

    def call(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv3(outputs)
        outputs = self.bn3(outputs)
        if self.first_layer:
            inputs = self.short_cut(inputs)
        d = self.out_planes
        outputs = tf.concat([inputs[:, :, :, :d] + outputs[:, :, :, :d], inputs[:, :, :, d:], outputs[:, :, :, d:]],
                            axis=-1)
        outputs = tf.nn.relu(outputs)
        return outputs


class DPN(keras.Model):
    def __init__(self, cfg, num_classes=10, weight_decay=1e-4):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.bn1 = layers.BatchNormalization()
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)

        self.pooling = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_classes,
                                  activation='softmax',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=keras.regularizers.l2(weight_decay))

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        sequential = keras.Sequential()
        strides = [stride] + [1] * (num_blocks - 1)
        for i, stride in enumerate(strides):
            sequential.add(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i == 0))
            self.last_planes = out_planes + (i + 2) * dense_depth
        return sequential

    def call(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.pooling(outputs)
        outputs = self.flatten(outputs)
        outputs = self.dense(outputs)
        return outputs


def DPN26(input_shape, num_classes=10, weight_decay=1e-4):
    cfg = {
        'in_planes': (96, 192, 384, 768),
        'out_planes': (256, 512, 1024, 2048),
        'num_blocks': (2, 2, 2, 2),
        'dense_depth': (16, 32, 24, 128)
    }
    inputs = keras.Input(shape=input_shape)
    outputs = DPN(cfg, num_classes=num_classes, weight_decay=weight_decay)(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)


def DPN92(input_shape, num_classes=10, weight_decay=1e-4):
    cfg = {
        'in_planes': (96, 192, 384, 768),
        'out_planes': (256, 512, 1024, 2048),
        'num_blocks': (3, 4, 20, 3),
        'dense_depth': (16, 32, 24, 128)
    }
    inputs = keras.Input(shape=input_shape)
    outputs = DPN(cfg, num_classes=num_classes, weight_decay=weight_decay)(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)
