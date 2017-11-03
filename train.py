# -*- coding: utf-8 -*-
from keras.utils import np_utils
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from generator import binarylab, gray2rgb, data_gen_small
import os
import numpy as np
import pandas as pd
import argparse
import json

from PSPNet import PSPNet50

if __name__ == "__main__":
    # command line argments
    parser = argparse.ArgumentParser(description="SegUNet LIP dataset")
    parser.add_argument("--train_list",
            default="../LIP/TrainVal_images/train_id.txt",
            help="train list path")
    parser.add_argument("--trainimg_dir",
            default="../LIP/TrainVal_images/TrainVal_images/train_images/",
            help="train image dir path")
    parser.add_argument("--trainmsk_dir",
            default="../LIP/TrainVal_parsing_annotations/TrainVal_parsing_annotations/train_segmentations/",
            help="train mask dir path")
    parser.add_argument("--val_list",
            default="../LIP/TrainVal_images/val_id.txt",
            help="val list path")
    parser.add_argument("--valimg_dir",
            default="../LIP/TrainVal_images/TrainVal_images/val_images/",
            help="val image dir path")
    parser.add_argument("--valmsk_dir",
            default="../LIP/TrainVal_parsing_annotations/TrainVal_parsing_annotations/val_segmentations/",
            help="val mask dir path")
    parser.add_argument("--batch_size",
            default=5,
            type=int,
            help="batch size")
    parser.add_argument("--n_epochs",
            default=50,
            type=int,
            help="number of epoch")
    parser.add_argument("--epoch_steps",
            default= 1000,
            type=int,
            help="number of epoch step")
    parser.add_argument("--val_steps",
            default=10,
            type=int,
            help="number of valdation step")
    parser.add_argument("--n_labels",
            default=20,
            type=int,
            help="Number of label")
    parser.add_argument("--input_shape",
            default=(512, 512, 3),
            help="Input images shape")
    parser.add_argument("--loss",
            default="categorical_crossentropy",
            type=str,
            help="loss function")
    parser.add_argument("--optimizer",
            default="adadelta",
            type=str,
            help="oprimizer")
    args = parser.parse_args()

    # set the necessary list
    train_list = pd.read_csv(args.train_list,header=None)
    val_list = pd.read_csv(args.val_list,header=None)

    # set the necessary directories
    trainimg_dir = args.trainimg_dir
    trainmsk_dir = args.trainmsk_dir
    valimg_dir = args.valimg_dir
    valmsk_dir = args.valmsk_dir

    # get old session
    old_session = KTF.get_session()

    with tf.Graph().as_default():
        session = tf.Session('')
        KTF.set_session(session)
        KTF.set_learning_phase(1)

        # set generater
        train_gen = data_gen_small(trainimg_dir,
                trainmsk_dir,
                train_list,
                args.batch_size,
                [args.input_shape[0], args.input_shape[1]],
                args.n_labels)
        val_gen = data_gen_small(valimg_dir,
                valmsk_dir,
                val_list,
                args.batch_size,
                [args.input_shape[0], args.input_shape[1]],
                args.n_labels)

        # set model
        pspnet = PSPNet50()
        print(pspnet.summary())

        # set callbacks
        fpath = "./pretrained/LIP_PSPNet50_{epoch:02d}.hdf5"
        cp_cb = ModelCheckpoint(filepath = fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=10)
        es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        tb_cb = TensorBoard(log_dir="./pretrained", write_images=True)

        # compile model
        pspnet.compile(loss=args.loss,
                optimizer=args.optimizer,
                metrics=["accuracy"])

        # fit with genarater
        pspnet.fit_generator(generator=train_gen,
                steps_per_epoch=args.epoch_steps,
                epochs=args.n_epochs,
                validation_data=val_gen,
                validation_steps=args.val_steps,
                callbacks=[cp_cb, es_cb, tb_cb])

        # save weights
        pspnet.save_weights("./pretrained/LIP_PSPNet50_stoped.hdf5")

    # save model
    with open("./pretrained/LIP_PSPNet50.json", "w") as json_file:
        json_file.write(json.dumps(json.loads(pspnet.to_json()), indent=2))
    print("save json model done...")
