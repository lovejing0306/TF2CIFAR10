# TF2CIFAR10

## Introduction
I trained CIFAR10 dataset with TensorFlow2, it is very easy to build the project environment by using TensorFlow2's docker container.
You can download it form [here](https://hub.docker.com/r/tensorflow/tensorflow/tags?page=1&name=2.0.0) .

If you find this project useful and using it in your work please cite this implementation, thanks.

## Requirements
* TensorFlow2.0
* Python3
* tqdm

## Accuracy
|Model|Acc|
| --- | --- |
| Resnet56  |94.0%|
| Resnet110  |94.6%|
| Resnet164  |94.6%|
| MobileNetV2  |94.3%|
| DPN92  |95.7%|

## Usage

1. Download CIFAR10 dataset in [Google Drive](https://drive.google.com/open?id=11vY3RJAp_4FC5mAx26q37ncOQRw3miE9) or [Baidu Yun](https://pan.baidu.com/s/1yvtaX628_EuKcjvXmHbOuQ) 85r9 and 
unzip it. Then, put them under directory
    ```angular2
    ./cifar/original
    ```
2. Convert CIFAR10 dataset by run `parse.py`
    ```angular2
    python3 ./dataset/parse.py
    ```
3. Train the model using `tain.py`, you need to modify `train.py` to choose a net you want to train.
    ```angular2
    python3 train.py
    ```
## Details of training
* optimizer:SGD
* lr:1e-1
* lr range:[(150, 1e-2), (250, 1e-3), (350, 1e-4)]
* weight decay:5e-4
* label smoothing:2e-1

Please read the `config.py` for more details.

## Pretrained models for download
You can test the accuracy of CIFAR10 by my models.

|Model|Link|
| --- | --- |
| Resnet56  |[Google Drive](https://drive.google.com/open?id=1DGC5aFPBEK-HFrCYjnFLK1xgjFtDzxA0) [Baidu Yun](https://pan.baidu.com/s/1GbZJ4oF7Vo6NTQ13B8D2VA) 477p|
| Resnet110  |[Google Drive](https://drive.google.com/open?id=14XYsN2S2MMl6_7pRqmK2LHm4i6BjuKoR) [Baidu Yun](https://pan.baidu.com/s/1jWL8LRSRrZXZ66c5y2MZiA) exus|
| Resnet164  |[Google Drive](https://drive.google.com/open?id=1nPgmTxOi85DLDWqjQISxhC7z6Z7HL0px) [Baidu Yun](https://pan.baidu.com/s/1oFpiKjLv1FfiXlalkOz5Jg) hy37|
| MobileNetV2  |[Google Drive](https://drive.google.com/open?id=1TW6PTPz7X2DnR88YdXhOGtSNqNb_r6Ep) [Baidu Yun](https://pan.baidu.com/s/1t7iBLkS-OoOOwmqO3pCfIw) 1pnj|
| DPN92  |[Google Drive](https://drive.google.com/open?id=1R-sGlJpBxzWi3ACgmUvZ4n5iHcdMLLHJ) [Baidu Yun](https://pan.baidu.com/s/1Jk9TCBvNJpHgojEFEwu1ug) 9ruw|

## Reference
[Keras-DualPathNetworks](https://github.com/titu1994/Keras-DualPathNetworks)

[pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

[Trains a ResNet on the CIFAR10 dataset](https://keras.io/examples/cifar10_resnet/)