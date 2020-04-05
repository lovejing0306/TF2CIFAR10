# coding=utf-8

from tensorflow import keras
from dataset.generator import generator
from callbacks import cb_learning_rate
from tqdm import tqdm
from nets import resnet
from nets import dpn
from nets import mobilenet_v2
import tensorflow as tf
import config
import datetime
import numpy as np
import os


class Trainer(object):
    def __init__(self, model_dir, summary_dir, model, optimizer, lr_range=None, label_smoothing=0.2):
        self.model_dir = model_dir
        self.summary_dir = summary_dir
        self.model = model
        self.optimizer = optimizer
        self.lr_range = lr_range
        self.label_smoothing = label_smoothing
        self.cb_lr = self.load_cb_lr()
        self.summary_writer = self.get_summary_writer()
        self.create_model_dir()

    def load_cb_lr(self):
        return cb_learning_rate.LearningRate(
            optimizer=self.optimizer,
            method=cb_learning_rate.RANGE,
            lr_range=self.lr_range
        )

    def create_model_dir(self):
        if self.model_dir is None:
            return None
        self.model_dir = os.path.join(self.model_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        os.makedirs(self.model_dir)

    def get_summary_writer(self):
        if self.summary_dir is None:
            return None
        else:
            log_dir = os.path.join(self.summary_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
            return tf.summary.create_file_writer(logdir=log_dir)

    def update_summary(self, **kwargs):  # 学习此种写法
        if self.summary_writer is None:
            pass
        else:
            with self.summary_writer.as_default():
                for name in kwargs:
                    tf.summary.scalar(name, kwargs[name], step=self.optimizer.iterations)

    def save_weights(self, filepath, save_format=None):
        self.model.save_weights(filepath=filepath, save_format=save_format)

    def load_weights(self, filepath):
        self.model.load_weights(filepath=filepath)

    @tf.function
    def accuracy(self, y_true, y_pred):
        res = keras.metrics.categorical_accuracy(y_true, y_pred)
        acc = tf.reduce_mean(res)
        sum = tf.reduce_sum(res)
        num = res.shape[0]
        return num, sum, acc

    @tf.function
    def train_on_batch(self, x, y):
        keras.backend.set_learning_phase(1)
        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = tf.math.reduce_mean(keras.losses.categorical_crossentropy(y_true=y,
                                                                             y_pred=y_pred,
                                                                             label_smoothing=self.label_smoothing)
                                       )
            if len(self.model.losses) == 0:
                total_loss = loss
            else:
                regularization_loss = tf.math.add_n(self.model.losses)
                total_loss = loss + regularization_loss
        variables = self.model.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.update_summary(loss=loss)
        return total_loss, loss, y_pred

    def train_on_epoch(self, epochs, epoch, dataset, show_every_steps):
        total_n = 0
        total_s = 0
        losses = list()
        stats = dict()
        pbar = tqdm(dataset)
        pbar.set_description('Epoch %03d / %03d' % (epoch, epochs))
        stats['lr'] = self.optimizer.learning_rate.numpy()
        for x, y in pbar:
            total_loss, loss, y_pred = self.train_on_batch(x, y)
            n, s, acc = self.accuracy(y_true=y, y_pred=y_pred)
            total_n += n.numpy()
            total_s += s.numpy()
            stats['iterations'] = self.optimizer.iterations.numpy()
            stats['loss'] = loss.numpy()
            stats['total loss'] = total_loss.numpy()
            losses.append(total_loss.numpy())
            if self.optimizer.iterations % show_every_steps == 0:
                avg_acc = total_s / (total_n + 1e-5)
                avg_loss = np.mean(losses)
                stats['avg loss'] = avg_loss
                stats['train acc'] = avg_acc
                self.update_summary(avg_acc=avg_acc,
                                    avg_loss=avg_loss
                                    )
            pbar.set_postfix(stats)
        pbar.close()

    @tf.function
    def val_on_batch(self, x):
        keras.backend.set_learning_phase(0)
        y_pred = self.model(x)
        return y_pred

    def val_on_epoch(self, epochs, epoch, dataset):
        total_n = 0
        total_s = 0
        avg_acc = 0
        pbar = tqdm(dataset)
        pbar.set_description('Epoch %03d / %03d' % (epoch, epochs))
        for x, y in pbar:
            y_pred = self.val_on_batch(x)
            n, s, acc = self.accuracy(y_true=y, y_pred=y_pred)
            total_n += n.numpy()
            total_s += s.numpy()
            avg_acc = total_s / (total_n + 1e-5)
            pbar.set_postfix({'val acc': avg_acc})
        pbar.close()
        self.update_summary(val_acc=avg_acc)

    def train(self, train_dataset, val_dataset, show_every_steps, epochs, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            self.cb_lr(epoch=epoch)
            self.train_on_epoch(epochs, epoch, train_dataset, show_every_steps)
            self.val_on_epoch(epochs, epoch, val_dataset)
            if self.model_dir is None:
                pass
            else:
                self.save_weights(os.path.join(self.model_dir, 'snapshot-%d.h5') % epoch)


def main():
    train_generator = generator(dir=config.train_dir,
                                num_classes=config.num_classes,
                                is_shuffle=True,
                                is_horizontal_flip=True,
                                is_random_crop=True,
                                is_random_cutout=True)
    train_generator = tf.data.Dataset.from_generator(train_generator,
                                                     output_types=(tf.float32, tf.int32))
    train_generator = train_generator.batch(batch_size=config.batch_size)
    train_generator = train_generator.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_generator = generator(dir=config.test_dir, num_classes=config.num_classes)
    val_generator = tf.data.Dataset.from_generator(val_generator,
                                                   output_types=(tf.float32, tf.int32))
    val_generator = val_generator.batch(batch_size=config.batch_size)
    val_generator = val_generator.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = resnet.ResNet(version=config.resnet_version,
                          input_shape=(config.height, config.width, config.num_channels),
                          depth=config.resnet_depth,
                          num_classes=config.num_classes)

    # model = dpn.DPN92(input_shape=(config.height, config.width, config.num_channels),
    #                    num_classes=config.num_classes,
    #                    weight_decay=config.weight_decay)

    # model = mobilenet_v2.MobileNetV2(input_shape=(config.height, config.width, config.num_channels),
    #                                  num_classes=config.num_classes,
    #                                  weight_decay=config.weight_decay)

    trainer = Trainer(model_dir=config.model_dir,
                      summary_dir=config.summary_dir,
                      model=model,
                      optimizer=keras.optimizers.SGD(learning_rate=config.lr, momentum=0.9),
                      lr_range=config.lr_range,
                      label_smoothing=config.label_smoothing)
    trainer.train(train_dataset=train_generator,
                  val_dataset=val_generator,
                  show_every_steps=config.show_every_steps,
                  epochs=config.epochs,
                  start_epoch=config.start_epoch)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
