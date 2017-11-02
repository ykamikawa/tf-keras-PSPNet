# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Multiply, Concatenate
from keras.utils import np_utils
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from os.path import splitext, join, isfile
from os import environ
from math import ceil
import argparse
import numpy as np
from scipy import misc, ndimage



