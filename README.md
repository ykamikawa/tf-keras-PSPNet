# keras-PSPNet

Pyramid Scene persing Network is a model of semantic segmentation based on Fully Comvolutional Network.
This repository contains the implementation of learning and testing in keras and tensorflow.


## Architecutre

- atrous convalution
- residual module
- pyramid pooling module

## Prerequirements

- python3.6
- opencv for python
- keras,tensorflow

## Usage

### train
- Segmentation involveing multiple categories

    ` python train.py --options `

- Segmentation of mask image

    ` python train_mask.py --options `

    - options
        - image dir
        - mask image dir
        - batchsize, nb_epochs, epoch_per_steps, input_configs
        - class weights
        - device num

### test
- Input test image
- responce json format involving category name and color(pixel based prediction)

    ` python predict.py --input_path [path/to//input_imahge] `

## DEMO

### dataset

- look into person

![demo1](https://user-images.githubusercontent.com/27678705/33703457-8a504fdc-db6b-11e7-8922-db3c61294b18.png)
