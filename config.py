# coding=utf-8

width = 32
height = 32
num_channels = 3

num_classes = 10
batch_size = 64
epochs = 400
show_every_steps = 100
start_epoch = 0

num_train_samples = 50000
lr = 1e-1
lr_range = [(150, 1e-2), (250, 1e-3), (350, 1e-4)]

weight_decay = 5e-4
label_smoothing = 0.2

model_name = 'mobilenet_v2'
resnet_version = 2
resnet_depth = 164

model_dir = './models/' + model_name + '/'
summary_dir = './summaries/' + model_name + '/'

train_dir = './cifar/train'
test_dir = './cifar/test'
