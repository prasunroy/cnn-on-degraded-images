# -*- coding: utf-8 -*-
"""
Training deep convolutional neural networks.
Created on Tue May 22 20:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/cnn-on-degraded-images

"""


# imports
from __future__ import division
from __future__ import print_function

import cv2
import glob
import numpy
import os
import platform
import tensorflow

from keras import applications
from keras import optimizers
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Model
from keras.utils import to_categorical
from matplotlib import pyplot
from scipy.io import loadmat

from mlutils.callbacks import Telegram


# configurations
# -----------------------------------------------------------------------------
PROCESS_SEED = None

ARCHITECTURE = 'inceptionv3'
INCLUDE_TOPL = False
WEIGHTS_INIT = 'imagenet'
INPUT_TENSOR = None
INPUT_DSHAPE = (299, 299, 3)
POOLING_TYPE = None
NUM_TCLASSES = 10
FREEZE_LAYER = 0
NEURONS_FC_1 = 1024
NEURONS_FC_2 = 1024
DROPOUT_FC12 = 0.5
FN_OPTIMIZER = optimizers.sgd(lr=0.0001, momentum=0.5)

DATASET_ID = 'synthetic_digits'
DATA_TRAIN = 'data/{}/data_train_299x299/data.mat'.format(DATASET_ID)
DATA_VALID = 'data/{}/data_valid_299x299/data.mat'.format(DATASET_ID)
IMAGE_SIZE = (299, 299, 3)
BATCH_SIZE = 50
NUM_EPOCHS = 100
OUTPUT_DIR = 'output/{}/{}/'.format(DATASET_ID, ARCHITECTURE)

AUTH_TOKEN = None
TELCHAT_ID = None

F_SHUTDOWN = False
# -----------------------------------------------------------------------------


# setup seed for random number generators for reproducibility
numpy.random.seed(PROCESS_SEED)
tensorflow.set_random_seed(PROCESS_SEED)


# setup paths for callbacks
log_dir = os.path.join(OUTPUT_DIR, 'logs')
cpt_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
tbd_dir = os.path.join(OUTPUT_DIR, 'tensorboard')

log_file = os.path.join(log_dir, 'training.csv')
cpt_best = os.path.join(cpt_dir, '{}_best.h5'.format(ARCHITECTURE))
cpt_last = os.path.join(cpt_dir, '{}_last.h5'.format(ARCHITECTURE))


# validate paths
def validate_paths():
    output_dirs = [OUTPUT_DIR, log_dir, cpt_dir, tbd_dir]
    for directory in output_dirs:
        if not os.path.isdir(directory):
            os.makedirs(directory)
        elif len(glob.glob(os.path.join(directory, '*.*'))) > 0:
            print('[INFO] Output directories must be empty')
            
            return False
    
    return True


# load data
def load_data():
    data_train = loadmat(DATA_TRAIN)['data']
    data_valid = loadmat(DATA_VALID)['data']
    
    numpy.random.shuffle(data_train)
    numpy.random.shuffle(data_valid)
    
    if INPUT_DSHAPE is None or IMAGE_SIZE == INPUT_DSHAPE:
        x_train = data_train[:, 1:].reshape((-1,) + INPUT_DSHAPE).astype('float32') / 255.0
        x_valid = data_valid[:, 1:].reshape((-1,) + INPUT_DSHAPE).astype('float32') / 255.0
    else:
        x_train = []
        x_valid = []
        
        for sample in data_train[:, 1:]:
            x = sample.reshape(IMAGE_SIZE)
            x = cv2.resize(x, INPUT_DSHAPE[:2])
            x_train.append(x)
        
        for sample in data_valid[:, 1:]:
            x = sample.reshape(IMAGE_SIZE)
            x = cv2.resize(x, INPUT_DSHAPE[:2])
            x_valid.append(x)
        
        x_train = numpy.asarray(x_train, dtype='float32') / 255.0
        x_valid = numpy.asarray(x_valid, dtype='float32') / 255.0
    
    y_train = to_categorical(data_train[:, 0]).astype('float32')
    y_valid = to_categorical(data_valid[:, 0]).astype('float32')
    
    return ((x_train, y_train), (x_valid, y_valid))


# build model
def build_model():
    model = None
    
    # create architecture
    if ARCHITECTURE.lower() == 'inceptionv3':
        model = applications.inception_v3.InceptionV3(include_top=INCLUDE_TOPL, weights=WEIGHTS_INIT, input_tensor=INPUT_TENSOR, input_shape=INPUT_DSHAPE, pooling=POOLING_TYPE, classes=NUM_TCLASSES)
    elif ARCHITECTURE.lower() == 'mobilenet':
        model = applications.mobilenet.MobileNet(include_top=INCLUDE_TOPL, weights=WEIGHTS_INIT, input_tensor=INPUT_TENSOR, input_shape=INPUT_DSHAPE, pooling=POOLING_TYPE, classes=NUM_TCLASSES)
    elif ARCHITECTURE.lower() == 'resnet50':
        model = applications.resnet50.ResNet50(include_top=INCLUDE_TOPL, weights=WEIGHTS_INIT, input_tensor=INPUT_TENSOR, input_shape=INPUT_DSHAPE, pooling=POOLING_TYPE, classes=NUM_TCLASSES)
    elif ARCHITECTURE.lower() == 'vgg16':
        model = applications.vgg16.VGG16(include_top=INCLUDE_TOPL, weights=WEIGHTS_INIT, input_tensor=INPUT_TENSOR, input_shape=INPUT_DSHAPE, pooling=POOLING_TYPE, classes=NUM_TCLASSES)
    elif ARCHITECTURE.lower() == 'vgg19':
        model = applications.vgg19.VGG19(include_top=INCLUDE_TOPL, weights=WEIGHTS_INIT, input_tensor=INPUT_TENSOR, input_shape=INPUT_DSHAPE, pooling=POOLING_TYPE, classes=NUM_TCLASSES)
    
    if not model is None:
        # freeze layers
        if FREEZE_LAYER < 0:
            for layer in model.layers:
                layer.trainable = False
        else:
            for layer in model.layers[:FREEZE_LAYER]:
                layer.trainable = False
        
        # add fully connected layers
        if not INCLUDE_TOPL:
            x = model.output
            x = Flatten()(x)
            x = Dense(NEURONS_FC_1, activation='relu')(x)
            x = Dropout(DROPOUT_FC12)(x)
            x = Dense(NEURONS_FC_2, activation='relu')(x)
            y = Dense(NUM_TCLASSES, activation='softmax')(x)
            
            # final architecture
            model_final = Model(inputs=model.input, outputs=y)
        else:
            model_final = model
        
        # compile the final model
        model_final.compile(optimizer=FN_OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model_final


# create callbacks
def callbacks():
    cb_log = CSVLogger(filename=log_file, append=True)
    cb_stp = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    cb_cpt_best = ModelCheckpoint(filepath=cpt_best, monitor='val_acc', save_best_only=True, verbose=1)
    cb_cpt_last = ModelCheckpoint(filepath=cpt_last, monitor='val_acc', save_best_only=False, verbose=0)
    cb_tbd = TensorBoard(log_dir=tbd_dir, batch_size=BATCH_SIZE, write_grads=True, write_images=True)
    cb_tel = Telegram(auth_token=AUTH_TOKEN, chat_id=TELCHAT_ID, monitor='val_acc', out_dir=OUTPUT_DIR)
    
    return [cb_log, cb_stp, cb_cpt_best, cb_cpt_last, cb_tbd, cb_tel]


# plot learning curve
def plot(train_history):
    # plot training and validation loss
    pyplot.figure()
    pyplot.plot(train_history.history['loss'], label='loss')
    pyplot.plot(train_history.history['val_loss'], label='val_loss')
    pyplot.title('Training and Validation Loss')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.legend()
    pyplot.savefig(os.path.join(log_dir, 'plot_loss.png'))
    pyplot.show()
    
    # plot training and validation accuracy
    pyplot.figure()
    pyplot.plot(train_history.history['acc'], label='acc')
    pyplot.plot(train_history.history['val_acc'], label='val_acc')
    pyplot.title('Training and Validation Accuracy')
    pyplot.xlabel('epoch')
    pyplot.ylabel('accuracy')
    pyplot.legend()
    pyplot.savefig(os.path.join(log_dir, 'plot_accuracy.png'))
    pyplot.show()
    
    return


# shutdown system
def shutdown():
    print('[INFO] Initiating system shutdown... ', end='')
    if platform.system() == 'Windows':
        flag = os.system('shutdown -s -t 0')
    else:
        flag = os.system('shutdown -h now')
    if flag == 0:
        print('succeeded')
    else:
        print('failed')
    
    return


# train model
def train():
    # validate paths
    if not validate_paths():
        return
    
    # load data
    print('[INFO] Loading data... ', end='')
    ((x_train, y_train), (x_valid, y_valid)) = load_data()
    print('done')
    
    # build model
    print('[INFO] Building model... ', end='')
    model = build_model()
    if model is None:
        print('failed')
        return
    else:
        print('done')
        model.summary()
    
    # create callbacks
    cb_list = callbacks()
    
    # acknowledgement
    data = {'chat_id': TELCHAT_ID,
            'text': '`Received a new training request.\nMODEL  : {}\nDATASET: {}`'\
            .format(ARCHITECTURE.upper(), DATASET_ID.upper()),
            'parse_mode': 'Markdown'}
    cb_list[-1]._send_message(data)
    
    # train model
    train_history = model.fit(x_train, y_train,
                              batch_size=BATCH_SIZE,
                              epochs=NUM_EPOCHS,
                              callbacks=cb_list,
                              validation_data=(x_valid, y_valid))
    
    # plot learning curve
    plot(train_history)
    
    return


# main
if __name__ == '__main__':
    train()
    if F_SHUTDOWN:
        shutdown()
