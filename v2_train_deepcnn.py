# -*- coding: utf-8 -*-
"""
Training deep convolutional neural networks.
Created on Mon Aug 26 14:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/cnn-on-degraded-images

"""


# imports
from __future__ import division
from __future__ import print_function

import glob
import json
import numpy
import os
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
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

from libs.PipelineUtils import shutdown
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
DATA_TRAIN = 'data/{}/imgs_train/'.format(DATASET_ID)
DATA_VALID = 'data/{}/imgs_valid/'.format(DATASET_ID)
LABEL_MAPS = 'data/{}/labelmap.json'.format(DATASET_ID)
SAVE_AUGMT = False
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


# setup paths for augmented data
if SAVE_AUGMT:
    aug_dir_train = os.path.join(OUTPUT_DIR, 'augmented_data/imgs_train/')
    aug_dir_valid = os.path.join(OUTPUT_DIR, 'augmented_data/imgs_valid/')
else:
    aug_dir_train = None
    aug_dir_valid = None


# setup paths for model architecture
mdl_dir = os.path.join(OUTPUT_DIR, 'models')
mdl_file = os.path.join(mdl_dir, '{}.json'.format(ARCHITECTURE))


# setup paths for callbacks
log_dir = os.path.join(OUTPUT_DIR, 'logs')
cpt_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
tbd_dir = os.path.join(OUTPUT_DIR, 'tensorboard')

log_file = os.path.join(log_dir, 'training.csv')
cpt_best = os.path.join(cpt_dir, '{}_best.h5'.format(ARCHITECTURE))
cpt_last = os.path.join(cpt_dir, '{}_last.h5'.format(ARCHITECTURE))


# validate paths
def validate_paths():
    flag = True
    data_dirs = [DATA_TRAIN, DATA_VALID]
    for directory in data_dirs:
        if not os.path.isdir(directory):
            print('[INFO] Data directory not found at {}'.format(directory))
            flag = False
    output_dirs = [OUTPUT_DIR, aug_dir_train, aug_dir_valid, mdl_dir, log_dir, cpt_dir, tbd_dir]
    output_dirs = [directory for directory in output_dirs if directory is not None]
    for directory in output_dirs:
        if not os.path.isdir(directory):
            os.makedirs(directory)
        elif len(glob.glob(os.path.join(directory, '*.*'))) > 0:
            print('[INFO] Output directory {} must be empty'.format(directory))
            flag = False
    
    return flag


# load data
def load_data():
    # image data generator configuration for training data augmentation
    data_gen_train = ImageDataGenerator(featurewise_center=False,
                                        samplewise_center=False,
                                        featurewise_std_normalization=False,
                                        samplewise_std_normalization=False,
                                        zca_whitening=False,
                                        zca_epsilon=1e-06,
                                        rotation_range=0.0,
                                        width_shift_range=0.0,
                                        height_shift_range=0.0,
                                        brightness_range=None,
                                        shear_range=0.0,
                                        zoom_range=0.0,
                                        channel_shift_range=0.0,
                                        fill_mode='nearest',
                                        cval=0.0,
                                        horizontal_flip=False,
                                        vertical_flip=False,
                                        rescale=1.0/255.0,
                                        preprocessing_function=None,
                                        data_format=None,
                                        validation_split=0.0)
    
    # image data generator configuration for validation data augmentation
    data_gen_valid = ImageDataGenerator(featurewise_center=False,
                                        samplewise_center=False,
                                        featurewise_std_normalization=False,
                                        samplewise_std_normalization=False,
                                        zca_whitening=False,
                                        zca_epsilon=1e-06,
                                        rotation_range=0.0,
                                        width_shift_range=0.0,
                                        height_shift_range=0.0,
                                        brightness_range=None,
                                        shear_range=0.0,
                                        zoom_range=0.0,
                                        channel_shift_range=0.0,
                                        fill_mode='nearest',
                                        cval=0.0,
                                        horizontal_flip=False,
                                        vertical_flip=False,
                                        rescale=1.0/255.0,
                                        preprocessing_function=None,
                                        data_format=None,
                                        validation_split=0.0)
    
    # training image data generator
    data_flow_train = data_gen_train.flow_from_directory(directory=DATA_TRAIN,
                                                         target_size=INPUT_DSHAPE[:-1],
                                                         color_mode='rgb',
                                                         classes=None,
                                                         class_mode='categorical',
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         seed=PROCESS_SEED,
                                                         save_to_dir=aug_dir_train,
                                                         save_prefix='',
                                                         save_format='png',
                                                         follow_links=False,
                                                         subset=None,
                                                         interpolation='nearest')
    
    # validation image data generator
    data_flow_valid = data_gen_valid.flow_from_directory(directory=DATA_VALID,
                                                         target_size=INPUT_DSHAPE[:-1],
                                                         color_mode='rgb',
                                                         classes=None,
                                                         class_mode='categorical',
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         seed=PROCESS_SEED,
                                                         save_to_dir=aug_dir_valid,
                                                         save_prefix='',
                                                         save_format='png',
                                                         follow_links=False,
                                                         subset=None,
                                                         interpolation='nearest')
    
    return (data_flow_train, data_flow_valid)


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
    cb_cpt_best = ModelCheckpoint(filepath=cpt_best, monitor='val_acc', save_best_only=True, save_weights_only=True, verbose=1)
    cb_cpt_last = ModelCheckpoint(filepath=cpt_last, monitor='val_acc', save_best_only=False, save_weights_only=True, verbose=0)
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
    pyplot.show(block=False)
    
    # plot training and validation accuracy
    pyplot.figure()
    pyplot.plot(train_history.history['acc'], label='acc')
    pyplot.plot(train_history.history['val_acc'], label='val_acc')
    pyplot.title('Training and Validation Accuracy')
    pyplot.xlabel('epoch')
    pyplot.ylabel('accuracy')
    pyplot.legend()
    pyplot.savefig(os.path.join(log_dir, 'plot_accuracy.png'))
    pyplot.show(block=False)
    
    return


# train model
def train():
    # validate paths
    if not validate_paths():
        return
    
    # load data
    (data_flow_train, data_flow_valid) = load_data()
    
    # save labelmap
    with open(LABEL_MAPS, 'w') as file:
        json.dump(data_flow_train.class_indices, file)
    print('[INFO] Created labelmap')
    
    # build model
    print('[INFO] Building model... ', end='')
    model = build_model()
    if model is None:
        print('failed')
        return
    else:
        print('done')
        model.summary()
    
    # serialize model to json
    model_json = model.to_json()
    with open(mdl_file, 'w') as file:
        file.write(model_json)
    
    # create callbacks
    cb_list = callbacks()
    
    # acknowledgement
    data = {'chat_id': TELCHAT_ID,
            'text': '`Received a new training request.\nTASK ID: {}\nMODEL  : {}\nDATASET: {}`'\
            .format(cb_list[-1]._task_id, ARCHITECTURE.upper(), DATASET_ID.upper()),
            'parse_mode': 'Markdown'}
    cb_list[-1]._send_message(data)
    
    # train model
    train_history = model.fit_generator(generator=data_flow_train,
                                        steps_per_epoch=None,
                                        epochs=NUM_EPOCHS,
                                        verbose=1,
                                        callbacks=cb_list,
                                        validation_data=data_flow_valid,
                                        validation_steps=None,
                                        class_weight=None,
                                        max_queue_size=10,
                                        workers=1,
                                        use_multiprocessing=False,
                                        shuffle=True,
                                        initial_epoch=0)
    
    # plot learning curve
    plot(train_history)
    
    return


# main
if __name__ == '__main__':
    train()
    if F_SHUTDOWN:
        shutdown()
