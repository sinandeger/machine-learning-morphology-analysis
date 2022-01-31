import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import sys
"""Import the basics: numpy, pandas, matplotlib, etc."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tqdm
import pickle
"""Import keras and other ML tools"""
import tensorflow as tf
import keras
print("keras version:", keras.__version__)
print("tensorflow version:", tf.__version__)
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization,\
    Flatten, Conv2D, AveragePooling2D, Add, ReLU, MaxPooling2D, GlobalAveragePooling2D, Concatenate
from keras.layers.core import Dropout, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import TensorBoard
from keras.regularizers import l2
"""Import scikit learn tools"""
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


"""
This module contains the functions employed to build an instance of DenseNet (Huang et al. 2016). There are differences
between this implementation and the structure proposed in the paper, e.g. there are no pooling layers in this build, 
as the network has been built to work with Hubble Space Telescope (HST) imaging cutouts of relatively small size (31x31 pixels). 

"""


def bn_relu_conv_block(input_, filter_count, weight_decay, kernel_dim=3, dropout_layer=False):
    """This function implements the Batch normalization - ReLU - Convolution block sequence"""

    bn_layer = BatchNormalization(axis=-1,
                                  gamma_regularizer=l2(weight_decay),
                                  beta_regularizer=l2(weight_decay))(input_)
    relu_act = Activation("relu")(bn_layer)
    conv_layer = Conv2D(kernel_size=kernel_dim, filters=filter_count, padding='same', kernel_regularizer=l2(weight_decay),
                        data_format="channels_last")(relu_act)

    if dropout_layer:
        conv_layer = Dropout(rate=0.20)(conv_layer)

    return conv_layer


def transition_layer(input_, filter_count, weight_decay, kernel_dim):
    """This function implements the transition layer"""

    bn_layer = BatchNormalization(axis=-1,
                                  gamma_regularizer=l2(weight_decay),
                                  beta_regularizer=l2(weight_decay))(input_)
    relu_act = Activation("relu")(bn_layer)
    conv_layer = Conv2D(strides=(2, 2), kernel_size=kernel_dim, filters=filter_count, kernel_regularizer=l2(weight_decay),
                        data_format="channels_last")(relu_act)

    return conv_layer


def dense_block(input_, num_layers, num_filters, growth_rate, weight_dec):
    """This function implements a complete denseblock, and performs the concatenation of features"""

    for layer_ind in range(num_layers):
        concat_vector = bn_relu_conv_block(input_, growth_rate, weight_decay=weight_dec, kernel_dim=3, dropout_layer=True)
        input_ = Concatenate(axis=-1)([input_, concat_vector])
        num_filters += growth_rate

    return input_, num_filters


def densenet_architecture(input_shape, blocks, num_filters, num_classes, num_conv_layers, growth_rate, weight_decay):

    """

    This is the function where we implement the full densenet, with the help of blocks we defined above.

    :param input_shape: The shape of the input image
    :param blocks: How many dense blocks to be assembled
    :param num_filters: The starting number of filters

    """

    """First, the initial convolutional block"""

    inputs = Input(shape=input_shape)

    x = Conv2D(num_filters, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(weight_decay),
               data_format="channels_last")(inputs)

    for block_ind in range(blocks):

        print('Dense block '+str(block_ind))

        x, num_filters = dense_block(x, num_layers=num_conv_layers, num_filters=num_filters,
                                     growth_rate=growth_rate, weight_dec=weight_decay)
        """Transition layer the follows dense blocks"""
        x = transition_layer(x, filter_count=num_filters, weight_decay=weight_decay, kernel_dim=1)

    """Append the last dense block here, not followed by a transition layer"""

    x, num_filters = dense_block(x, num_layers=num_conv_layers, num_filters=num_filters,
                                 growth_rate=growth_rate, weight_dec=weight_decay)

    x = BatchNormalization(axis=-1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = Flatten()(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(rate=0.25)(x)
    output = Dense(num_classes, activation='softmax')(x)

    densenet_model = Model(inputs, output)

    return densenet_model


def fit_densenet(train_imgs, train_labels, test_imgs, test_labels, num_classes, class_names,
                 densenet_specs_dict):

    """
    This function builds the densenet as specified by densenet_specs_dict, and fits it


    densenet_specs_dict needs to specify the number of filters at the beginning layer of the network, the number of blocks,
    the number of convolutional layers inside the blocks, and the growth rate

    """

    """DenseNet identifier tag"""
    densenet_model_tag = 'b'+str(densenet_specs_dict['blocks'])+'f'+str(densenet_specs_dict['num_filters'])+\
                         'l'+str(densenet_specs_dict['num_conv_layers'])+'g'+str(densenet_specs_dict['growth_rate'])

    densenet_model = densenet_architecture(input_shape=train_imgs.shape[1:], blocks=densenet_specs_dict['blocks'],
                                           num_filters=densenet_specs_dict['num_filters'],
                                           num_classes=num_classes, num_conv_layers=densenet_specs_dict['num_conv_layers'],
                                           growth_rate=densenet_specs_dict['growth_rate'],
                                           weight_decay=densenet_specs_dict['weight_decay'])

    # adam_optimizer = keras.optimizers.adam(lr=0.0001)
    sgd_optimizer = keras.optimizers.SGD(lr=0.005, momentum=0.0, nesterov=False)
    densenet_model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    """Set the specifications of the data augmentation to be performed below"""
    datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                 samplewise_center=False,  # set each sample mean to 0
                                 featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                 samplewise_std_normalization=False,  # divide each input by its std
                                 zca_whitening=False,  # apply ZCA whitening
                                 rotation_range=0.0,  # randomly rotate images in the range (degrees, 0 to 180)
                                 width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
                                 height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
                                 horizontal_flip=True,  # randomly flip images
                                 vertical_flip=True)  # randomly flip images

    datagen.fit(train_imgs)

    """Fit the model on the batches generated by datagen.flow()."""
    batch_size = 32
    epochs = 150
    classifier = densenet_model.fit_generator(datagen.flow(train_imgs, train_labels, batch_size=batch_size),
                                              steps_per_epoch=train_imgs.shape[0] // batch_size,
                                              validation_data=(test_imgs, test_labels),
                                              epochs=epochs, verbose=1, max_q_size=100, shuffle=True)

    """Plot accuracy/loss versus epoch"""
    fig = plt.figure(figsize=(10, 3))

    ax1 = plt.subplot(121)
    ax1.plot(classifier.history['accuracy'], color='darkslategray', linewidth=2, label='Accuracy')
    ax1.plot(classifier.history['val_accuracy'], color='forestgreen', linewidth=2, label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.xaxis.set_label_coords(0.5, -0.15)
    ax1.legend(loc='lower right')

    ax2 = plt.subplot(122)
    ax2.plot(classifier.history['loss'], color='crimson', linewidth=2, label='Loss')
    ax2.plot(classifier.history['val_loss'], color='silver', linewidth=2, label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.xaxis.set_label_coords(2.0, -0.15)
    ax2.set_ylim(0, 2)
    ax2.legend(loc='upper right')

    plt.savefig('densenet_'+densenet_model_tag+'_accuracy-loss.pdf', type='pdf')
    plt.close()

    preds_test = densenet_model.predict(test_imgs)
    preds_test_t = np.argmax(preds_test, axis=1)

    test_labels_t = np.argmax(test_labels, axis=1)

    print('Fraction of correct predictions with the DenseNet:',
          round(accuracy_score(test_labels_t, preds_test_t), 2))

    conf_classes = np.array(class_names)
    print(conf_classes.dtype)
    plot_confusion_matrix(test_labels_t, preds_test_t, normalize=True, classes=conf_classes,
                          title='Normalized Confusion Matrix -- DenseNet '+densenet_model_tag)
    plt.savefig(str(len(conf_classes)) + '_class-pred_confmatrix_densenet_'+densenet_model_tag+'.pdf', type='pdf')
    plt.close()


"""Original definition of the transition layer below"""
# def transition_layer(input_, filter_count, kernel_dim=1):
#
#     """This function implements the transition layer"""
#
#     bn_layer = BatchNormalization()(input_)
#     conv_layer = Conv2D(kernel_size=kernel_dim, filters=filter_count, strides=1, data_format="channels_last")(bn_layer)
#     avg_pool = AveragePooling2D((2, 2), strides=(2, 2))(conv_layer)
#
#     return avg_pool


# def dense_block(input_, num_layers, num_filters, growth_rate):
#
#     """This fuction implements a complete denseblock, and performs the concatenation of features"""
#
#     input_num_filters = num_filters
#
#     for layer_ind in range(num_layers):
#
#         concat_vector = bn_relu_conv_block(input_, input_num_filters, kernel_dim=3, dropout_layer=True)
#         input_ = Concatenate(axis=-1)([input_, concat_vector])
#         num_filters += growth_rate
#
#     return input_, num_filters








