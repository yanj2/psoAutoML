# Ensure backwards compatibility with Python 2
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals)
# from builtins import *
import logging
import platform
import datetime
import argparse
import time

import tensorflow as tf 

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

from sklearn import datasets
from sklearn.model_selection import train_test_split

# import dne.somMaps
# from dne.logger import LOGGER, set_up_logger

import numpy as np

# LOGGER = logging.getLogger(__name__)

NUM_CLASSES = 10 
NUM_ROWS = 8
NUM_COLS = 8
NUM_CHANNELS = 1

NUM_EPOCHS = 2
NUM_CONV_FILTERS = 32
KERNEL_SIZE = 3
POOLING_SIZE = 2
DROPOUT_RATE = 0.5
NUM_DENSE_OUT = 128

def train_model(nb_classes = NUM_CLASSES,
                img_rows = NUM_ROWS,
                img_cols = NUM_COLS,
                img_channels = NUM_CHANNELS,
                num_epochs = NUM_EPOCHS,
                num_conv_outputs = NUM_CONV_FILTERS,
                kernel_size = KERNEL_SIZE,
                pooling_size = POOLING_SIZE,
                dropout_rate = DROPOUT_RATE,
                num_dense_outputs = NUM_DENSE_OUT):
    # LOGGER.info("Assembling model components.")
    print("Assembling model components.")

    # file locking --> 

    with tf.device('/CPU:0'):

        model = build_model(nb_classes,
                            img_rows,
                            img_cols,
                            img_channels,
                            num_conv_outputs,
                            kernel_size,
                            pooling_size,
                            dropout_rate,
                            num_dense_outputs)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1.e-6, momentum=0.9, nesterov=True)
    # LOGGER.info("Compiling model.")
    print("Compiling model.")
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # LOGGER.info("Loading in the SOM maps.")
    # (X_train, y_train), (X_test, y_test) = somMaps.load_data()
    # # LOGGER.debug('X_train shape:', X_train.shape)
    # # LOGGER.debug(X_train.shape[0], 'train samples')
    # # LOGGER.debug(X_test.shape[0], 'test samples')

    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255

    # LOGGER.info("Loading data from npz")
    # print("Loadgin data from npz")
    # data = np.load("training_data.npz")
    data = datasets.load_digits()
    x = data.data
    y = data.target 
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

    # x_train = data['x_train']
    # y_train = data['y_train']
    # x_test = data['x_test']
    # y_test = data['y_test']

    # reshape the x_train, x_test from 1 dim to 2 
    x_train = np.reshape(x_train, (-1, NUM_ROWS, NUM_COLS, NUM_CHANNELS))
    x_test = np.reshape(x_test, (-1, NUM_ROWS, NUM_COLS, NUM_CHANNELS))

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    batch_size = 32
    data_augmentation = False

    # LOGGER.info("Training model.")
    print("Training model.")
    start = time.time()
    if not data_augmentation:

        # LOGGER.info('Not using data augmentation.')
        print("Not using data augmentation")
        hist = model.fit(x_train,
                         y_train,
                         batch_size=batch_size,
                         nb_epoch=num_epochs,
                         validation_data=(x_test, y_test),
                         shuffle=True)
    else:
        # LOGGER.info('Using real-time data augmentation.')
        print("Using real-time data augmentation")

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(featurewise_center=False,
                                     samplewise_center=False,
                                     featurewise_std_normalization=False,
                                     samplewise_std_normalization=False,
                                     zca_whitening=False,
                                     rotation_range=0,
                                     width_shift_range=0.2,
                                     height_shift_range=0.1,
                                     horizontal_flip=True,
                                     vertical_flip=False)

        # compute quantities required for featurwise normalization
        # (std, mean and principal components if ZCA whitening is applied)
        datagen.fit(x_train)

        # fit the model on the batches generated by datagen.flow()
        hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                   samples_per_epoch=x_train.shape[0],
                                   nb_epoch=num_epochs,
                                   validation_data=(x_test, y_test))

    end = time.time()

    # LOGGER.debug("%s", repr(hist.history))
    print("%s".format(repr(hist.history)))

    learning_time = end-start
    try:
        validation_accuracy = hist.history['val_acc'][-1]
    except KeyError:
        validation_accuracy = 0.
    print("{0}, {1}".format(learning_time, validation_accuracy))
    # printing these values out to the stdout and reading in all these values back.... 

    return validation_accuracy

def build_model(nb_classes,
                img_rows,
                img_cols,
                img_channels=3,
                num_conv_filters=32,     # 512 or 1024
                kernel_size=3,           # max image size: kernel x kernel < width x height 
                pooling_size=2,          # max image size: kernel x kernel < width x height no  
                dropout_rate=0.5,        # % value
                num_dense_outputs=128):  # 512 or 1024 n
    model = Sequential()

    model.add(Convolution2D(num_conv_filters,
                            kernel_size,
                            kernel_size,
                            border_mode='same',
                            input_shape=(img_rows, img_cols, img_channels),
                            subsample=(2,2)))
    model.add(Activation('relu'))
    # model.add(Convolution2D(num_conv_filters,
                            # kernel_size,
                            # kernel_size,
                            # subsample=(2,2)))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pooling_size, pooling_size)))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(num_dense_outputs))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model

def keras_capos_input_cli():
    """Command line interface to the keras CAPOS implementation.
    This function parses the command line functions and initialises the image
    processing pipeline.
    """
    parser = argparse.ArgumentParser(description="""A script for generating
                                                    the input files for digits.""")
    parser.add_argument("--num_classes",
                        default=NUM_CLASSES,
                        type=int,
                        help="Number of classes.")
    parser.add_argument("--num_rows",
                        default=NUM_ROWS,
                        type=int,
                        help="Number of rows in the input images.")
    parser.add_argument("--num_cols",
                        default=NUM_COLS,
                        type=int,
                        help="Number of columns in the input images.")
    parser.add_argument("--num_channels",
                        default=NUM_CHANNELS,
                        type=int,
                        help="Number of colour channels in the input images.")
    parser.add_argument("--num_epochs",
                        default=NUM_EPOCHS,
                        type=int,
                        help="Number of epochs")
    parser.add_argument("--num_conv_filters",
                        default=NUM_CONV_FILTERS,
                        type=int,
                        help="""Number of output filters from the first convolution layer.""")
    parser.add_argument("--kernel_size",
                        default=KERNEL_SIZE,
                        type=int,
                        help="Size of the convolutional filter.")
    parser.add_argument("--pooling_size",
                        default=POOLING_SIZE,
                        type=int,
                        help="Size of the pooling kernel.")
    parser.add_argument("--dropout_rate",
                        default=DROPOUT_RATE,
                        type=float,
                        help="Rate of dropout.")
    parser.add_argument("--num_dense_outputs",
                        default=NUM_DENSE_OUT,
                        type=int,
                        help="Number of outputs from the dense (fully connected) layer.")
    args = parser.parse_args()

    # set_up_logger()

    # LOGGER.info(" ")
    # LOGGER.info("*********************************************")
    # LOGGER.info("* KERAS CAPOS INPUT                         *")
    # LOGGER.info("*********************************************")
    # LOGGER.info(" ")
    # LOGGER.info("Input parameters are:")
    # LOGGER.info("Number of classes - %i", args.num_classes)
    # LOGGER.info("Number of rows - %i", args.num_rows)
    # LOGGER.info("Number of columns - %i", args.num_cols)
    # LOGGER.info("Number of channels - %i", args.num_channels)
    # LOGGER.info(" ")
    # LOGGER.info("---------------------------------------------")
    # LOGGER.info(" ")
    # LOGGER.info("Trainable parameters")
    # LOGGER.info(" ")
    # LOGGER.info("Number of epochs - %i", args.num_epochs)
    # LOGGER.info("Number of filters output from convolution layers- %i", args.num_conv_filters)
    # LOGGER.info("Size of convolutional kernel - (%i, %i)", args.kernel_size, args.kernel_size)
    # LOGGER.info("Size of pooling kernel - (%i, %i)", args.pooling_size, args.pooling_size)
    # LOGGER.info("Dropout rate - %f", args.dropout_rate)
    # LOGGER.info("Number of dense layer outputs - %i", args.num_dense_outputs)
    # LOGGER.info(" ")
    # LOGGER.info("*********************************************")
    # LOGGER.info(" ")

    print(" ")
    print("*********************************************")
    print("* KERAS CAPOS INPUT                         *")
    print("*********************************************")
    print(" ")
    print("Input parameters are:")
    print("Number of classes - {:d}".format(args.num_classes))
    print("Number of rows - {:d}".format(args.num_rows))
    print("Number of columns - {:d}".format(args.num_cols))
    print("Number of channels - {:d}".format(args.num_channels))
    print(" ")
    print("---------------------------------------------")
    print(" ")
    print("Trainable parameters")
    print(" ")
    print("Number of epochs - {:d}".format(args.num_epochs))
    print("Number of filters output from convolution layers- {:d}".format(args.num_conv_filters))
    print("Size of convolutional kernel - ({:d}, {:d})".format(args.kernel_size, args.kernel_size))
    print("Size of pooling kernel - ({:d}, {:d})".format(args.pooling_size, args.pooling_size))
    print("Dropout rate - {:f}".format(args.dropout_rate))
    print("Number of dense layer outputs - {:d}".format(args.num_dense_outputs))
    print(" ")
    print("*********************************************")
    print(" ")

    train_model(args.num_classes,
                args.num_rows,
                args.num_cols,
                args.num_channels,
                args.num_epochs,
                args.num_conv_filters,
                args.kernel_size,
                args.pooling_size,
                args.dropout_rate,
                args.num_dense_outputs)

# def keras_capos_input_pei(num_classes = NUM_CLASSES, 
#                           num_rows = NUM_ROWS, 
#                           num_cols = NUM_COLS, 
#                           num_channels = NUM_CHANNELS, 
#                           num_epochs = NUM_EPOCHS, 
#                           num_conv_filters = NUM_CONV_FILTERS,
#                           kernel_size = KERNEL_SIZE,
#                           pooling_size = POOLING_SIZE,
#                           dropout_rate = DROPOUT_RATE,
#                           num_dense_out = NUM_DENSE_OUT):
#     """ PSO evaluation interface. This function takes in hyperparameter values 
#     as arguments and initialises the image processing pipeline. 
#     """

#     print(" ")
#     print("*********************************************")
#     print("* KERAS CAPOS INPUT                         *")
#     print("*********************************************")
#     print(" ")
#     print("Input parameters are:")
#     print("Number of classes - {:d}".format(num_classes))
#     print("Number of rows - {:d}".format(num_rows))
#     print("Number of columns - {:d}".format(num_cols))
#     print("Number of channels - {:d}".format(num_channels))
#     print(" ")
#     print("---------------------------------------------")
#     print(" ")
#     print("Trainable parameters")
#     print(" ")
#     print("Number of epochs - {:d}".format(num_epochs))
#     print("Number of filters output from convolution layers- {:d}".format(num_conv_filters))
#     print("Size of convolutional kernel - ({:d}, {:d})".format(kernel_size, kernel_size))
#     print("Size of pooling kernel - ({:d}, {:d})".format(pooling_size, pooling_size))
#     print("Dropout rate - {:f}".format(dropout_rate))
#     print("Number of dense layer outputs - {:d}".format(num_dense_outputs))
#     print(" ")
#     print("*********************************************")
#     print(" ")

#     return train_model(num_classes,
#                 num_rows,
#                 num_cols,
#                 num_channels,
#                 num_epochs,
#                 num_conv_filters,
#                 kernel_size,
#                 pooling_size,
#                 dropout_rate,
#                 num_dense_out)
    



if __name__ == '__main__':
    keras_capos_input_cli()
