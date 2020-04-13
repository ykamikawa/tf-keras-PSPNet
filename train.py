import argparse
import os

import keras.backend.tensorflow_backend as KTF
import pandas as pd
import tensorflow as tf
from generator import data_gen_small
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from model import PSPNet50


def argparer():
    # command line argments
    parser = argparse.ArgumentParser(description="PSPNet LIP dataset")
    parser.add_argument("--train_list", help="train list path")
    parser.add_argument("--trainimg_dir", help="train image dir path")
    parser.add_argument("--trainmsk_dir", help="train mask dir path")
    parser.add_argument("--val_list", help="val list path")
    parser.add_argument("--valimg_dir", help="val image dir path")
    parser.add_argument("--valmsk_dir", help="val mask dir path")
    parser.add_argument("--batch_size", default=5, type=int, help="batch size")
    parser.add_argument("--n_epochs", default=10, type=int, help="number of epoch")
    parser.add_argument(
        "--epoch_steps", default=6000, type=int, help="number of epoch step"
    )
    parser.add_argument(
        "--val_steps", default=1000, type=int, help="number of valdation step"
    )
    parser.add_argument("--n_labels", default=20, type=int, help="Number of label")
    parser.add_argument(
        "--input_shape", default=(512, 512, 3), help="Input images shape"
    )
    parser.add_argument("--output_stride", default=16, type=int, help="output stirde")
    parser.add_argument(
        "--output_mode", default="softmax", type=str, help="output activation"
    )
    parser.add_argument(
        "--upsample_type", default="deconv", type=str, help="upsampling type"
    )
    parser.add_argument(
        "--loss", default="categorical_crossentropy", type=str, help="loss function"
    )
    parser.add_argument("--optimizer", default="adadelta", type=str, help="oprimizer")
    parser.add_argument("--gpu", default="0", type=str, help="number of gpu")
    args = parser.parse_args()

    return args


def main(args):
    # device number
    if args.gpu_num:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set the necessary list
    train_list = pd.read_csv(args.train_list, header=None)
    val_list = pd.read_csv(args.val_list, header=None)

    # set the necessary directories
    trainimg_dir = args.trainimg_dir
    trainmsk_dir = args.trainmsk_dir
    valimg_dir = args.valimg_dir
    valmsk_dir = args.valmsk_dir

    # get old session old_session = KTF.get_session()

    with tf.Graph().as_default():
        session = tf.Session("")
        KTF.set_session(session)
        KTF.set_learning_phase(1)

        # set callbacks
        cp_cb = ModelCheckpoint(
            filepath=args.log_dir,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="auto",
            period=2,
        )
        es_cb = EarlyStopping(monitor="val_loss", patience=2, verbose=1, mode="auto")
        tb_cb = TensorBoard(log_dir=args.log_dir, write_images=True)

        # set generater
        train_gen = data_gen_small(
            trainimg_dir,
            trainmsk_dir,
            train_list,
            args.batch_size,
            [args.input_shape[0], args.input_shape[1]],
            args.n_labels,
        )
        val_gen = data_gen_small(
            valimg_dir,
            valmsk_dir,
            val_list,
            args.batch_size,
            [args.input_shape[0], args.input_shape[1]],
            args.n_labels,
        )

        # set model
        pspnet = PSPNet50(
            input_shape=args.input_shape,
            n_labels=args.n_labels,
            output_mode=args.output_mode,
            upsample_type=args.upsample_type,
        )
        print(pspnet.summary())

        # compile model
        pspnet.compile(loss=args.loss, optimizer=args.optimizer, metrics=["accuracy"])

        # fit with genarater
        pspnet.fit_generator(
            generator=train_gen,
            steps_per_epoch=args.epoch_steps,
            epochs=args.n_epochs,
            validation_data=val_gen,
            validation_steps=args.val_steps,
            callbacks=[cp_cb, es_cb, tb_cb],
        )


if __name__ == "__main__":

    args = argparer()
    main(args)
