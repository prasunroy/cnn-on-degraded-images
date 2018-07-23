# -*- coding: utf-8 -*-
"""
Performance test of cnn architectures on various degradation models.
Created on Thu May 24 11:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/cnn-on-degraded-images

"""


# imports
from __future__ import division
from __future__ import print_function

import cv2
import glob
import json
import numpy
import os
import pandas
import sys

from keras.applications import mobilenet
from keras.models import load_model
from matplotlib import pyplot

from libs.CapsuleNetwork import CapsuleLayer
from libs.CapsuleNetwork import Mask
from libs.CapsuleNetwork import Length
from libs.DegradationModels import imdegrade
from libs.PipelineUtils import save_samples
from libs.PipelineUtils import shutdown
from mlutils.callbacks import Telegram


# configurations
# -----------------------------------------------------------------------------
RANDOM_SEED = None

RANDOM_STR = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PROCESS_ID = ''.join([RANDOM_STR[numpy.random.randint(0, len(RANDOM_STR))] \
                      for _ in range(16)])

DATASET_ID = 'synthetic_digits'
LABEL_MAPS = 'data/{}/labelmap.json'.format(DATASET_ID)
IMAGE_DSRC = 'data/{}/imgs_valid/'.format(DATASET_ID)
IMAGE_READ = 1
SAVE_NOISY = False
SAMP_NOISY = 10

NOISE_LIST = ['Gaussian_White', 'Gaussian_Color', 'Salt_and_Pepper',
              'Motion_Blur', 'Gaussian_Blur', 'JPEG_Quality']
MODEL_LIST = ['capsnet', 'inceptionv3', 'mobilenet', 'resnet50',
              'vgg16', 'vgg19']
MODEL_DICT = {name.lower(): 'output/{}/{}/checkpoints/{}_best.h5'\
              .format(DATASET_ID, name, name) for name in MODEL_LIST}
TOP_N_PRED = 3

OUTPUT_DIR_NOISY = 'output/{}/__test__images__'.format(DATASET_ID)
OUTPUT_DIR_TOP_1 = 'output/{}/__test__top{}__/'.format(DATASET_ID, 1)
OUTPUT_DIR_TOP_N = 'output/{}/__test__top{}__/'.format(DATASET_ID, TOP_N_PRED)

AUTH_TOKEN = None
TELCHAT_ID = None
TEL_CLIENT = Telegram(auth_token=AUTH_TOKEN, chat_id=TELCHAT_ID)

F_SHUTDOWN = False
# -----------------------------------------------------------------------------


# setup parameters
sigmavals = [x for x in range(0, 256, 5)]
densities = [x/100 for x in range(0, 101, 5)]
mb_ksizes = [x for x in range(3, 32, 2)]
gb_ksizes = [x for x in range(1, 52, 2)]
qualities = [x for x in range(30, -1, -2)]

# setup acknowledgement message templates
ack_msg_beg = """`
Received a new test request.

TASK ID: {}
DATASET: {}
SAMPLES: {}

Using CNN architectures:
{}
Using degradation models:
{}
`"""\
.format(PROCESS_ID,
        DATASET_ID,
        {},
        ''.join(['\t* {}\n'.format(model) for model in MODEL_LIST]),
        ''.join(['\t* {}\n'.format(noise) for noise in NOISE_LIST]))

ack_msg_end = """`
An ongoing test is finished.

TASK ID: {}
`"""\
.format(PROCESS_ID)

ack_msg_int = """`
An ongoing test is interrupted.

TASK ID: {}
REASONS: {}
`"""\
.format(PROCESS_ID, {})


# validate paths
def validate_paths():
    flag = True
    if not os.path.isfile(LABEL_MAPS):
        print('[INFO] Label mapping not found at {}'.format(LABEL_MAPS))
        flag = False
    if not os.path.isdir(IMAGE_DSRC):
        print('[INFO] Image data source not found at {}'.format(IMAGE_DSRC))
        flag = False
    for name, path in MODEL_DICT.items():
        if not os.path.isfile(path):
            print('[INFO] Model checkpoint not found at {}'.format(path))
            flag = False
    for directory in [OUTPUT_DIR_NOISY, OUTPUT_DIR_TOP_1, OUTPUT_DIR_TOP_N]:
        if not os.path.isdir(directory):
            os.makedirs(directory)
        elif len(glob.glob(os.path.join(directory, '*.*'))) > 0:
            print('[INFO] Output directory {} must be empty'.format(directory))
            flag = False
    
    return flag


# load data
def load_data():
    x = []
    y = []
    
    # label mapping
    with open(LABEL_MAPS, 'r') as file:
        labelmap = json.load(file)
    
    # class labels
    labels = [os.path.split(d[0])[-1] for d in os.walk(IMAGE_DSRC)][1:]
    
    # read images
    for label in labels:
        for file in glob.glob(os.path.join(IMAGE_DSRC, label, '*.*')):
            image = cv2.imread(file, IMAGE_READ)
            if image is None:
                continue
            x.append(image)
            y.append(labelmap[label])
    
    return (x, y)


# load models
def load_models():
    models = {}
    for name, path in MODEL_DICT.items():
        if name.lower() == 'capsnet':
            models[name] = load_model(path, custom_objects={'CapsuleLayer': CapsuleLayer,
                                                            'Mask': Mask,
                                                            'Length': Length})
        elif name.lower() == 'mobilenet':
            models[name] = load_model(path, custom_objects={'relu6': mobilenet.relu6})
        else:
            models[name] = load_model(path)
    
    return models


# initialize histories
def init_histories(init_dict={}):
    histories_top_1 = init_dict.copy()
    histories_top_n = init_dict.copy()
    for name in MODEL_LIST:
        histories_top_1['acc_' + name.lower()] = []
        histories_top_n['acc_' + name.lower()] = []
    
    return [histories_top_1, histories_top_n]


# save and plot histories
def save_and_plot_histories(file_id, histories, title='', xlabel='', ylabel='',
                            invert_xaxis=False, invert_yaxis=False):
    for hist_dict, output_dir, ylabel_prefix \
    in zip(histories,
           [OUTPUT_DIR_TOP_1, OUTPUT_DIR_TOP_N],
           ['', 'top-{} '.format(TOP_N_PRED)]):
        # save histories
        df = pandas.DataFrame(hist_dict)
        df.to_csv(os.path.join(output_dir, str(file_id) + '.csv'), index=False)
        
        # plot histories
        pyplot.figure()
        pyplot.title(title)
        pyplot.xlabel(xlabel)
        pyplot.ylabel(ylabel_prefix + ylabel)
        if invert_xaxis:
            pyplot.gca().invert_xaxis()
        if invert_yaxis:
            pyplot.gca().invert_yaxis()
        x = [key for key in hist_dict.keys() if key.split('_')[0] != 'acc'][0]
        for y in hist_dict.keys():
            if y == x:
                continue
            pyplot.plot(hist_dict[x], hist_dict[y], label=y.split('_')[-1])
        pyplot.legend()
        pyplot.savefig(os.path.join(output_dir, str(file_id) + '.png'))
        pyplot.show()
        
        # acknowledgement
        plot_cap = '`{} [TASK ID: {}]`'.format(title, PROCESS_ID)
        plot_img = open(os.path.join(output_dir, str(file_id) + '.png'), 'rb')
        
        TEL_CLIENT._send_photo(data={'chat_id': TELCHAT_ID,
                                     'caption': plot_cap,
                                     'parse_mode': 'Markdown'},
                               files={'photo': plot_img})
    
    return


# test models
def test_models(x, y, models):
    print('')
    samples = len(x)
    results_top_1 = {}
    results_top_n = {}
    for name in models.keys():
        print('[INFO] Preparing images for {}'.format(name))
        images = []
        counts = 0
        for image in x:
            images.append(cv2.resize(image, models[name].input_shape[1:3]))
            counts += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(counts*100/samples), end='')
        print('\n[INFO] Testing images on {}... '.format(name), end='')
        x_test = numpy.asarray(images, dtype='float32') / 255.0
        y_test = numpy.asarray(y, dtype='int')
        if name == 'capsnet':
            p_test = models[name].predict(x_test)[0].argsort(axis=1)[:, -TOP_N_PRED:]
        else:
            p_test = models[name].predict(x_test).argsort(axis=1)[:, -TOP_N_PRED:]
        accuracy_top_1 = sum(p_test[:, -1]==y_test) * 100.0 / samples
        accuracy_top_n = sum([int(y in p) for y, p in zip(y_test, p_test)]) * 100.0 / samples
        results_top_1['acc_' + name] = accuracy_top_1
        results_top_n['acc_' + name] = accuracy_top_n
        print('done\t\t[accuracy: {:6.2f}%] [top-{} accuracy: {:6.2f}%]'\
              .format(accuracy_top_1, TOP_N_PRED, accuracy_top_n))
    
    return [results_top_1, results_top_n]


# tests for gaussian white noise
def test_gaussian_white(x, y, models):
    # initialize histories
    histories = init_histories({'sigma': sigmavals})
    
    # run tests
    for sigma in sigmavals:
        # apply gaussian white noise
        print('[INFO] Applying Gaussian white noise with mu=0 and sigma={}'\
              .format(sigma))
        noisy = []
        count = 0
        for image in x:
            noisy.append(imdegrade(image, 'gaussian_white', mu=0, sigma=sigma,
                                   seed=RANDOM_SEED))
            count += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(count*100/len(x)), end='')
        
        # save noisy image samples if required
        if SAVE_NOISY:
            save_samples(noisy, SAMP_NOISY, randomize=True, seed=RANDOM_SEED,
                         filename_prefix='gaussian_white_sigma_{}'.format(sigma),
                         target_directory=OUTPUT_DIR_NOISY)
        
        # test models
        results = test_models(noisy, y, models)
        
        # update histories
        for hist_dict, res_dict in zip(histories, results):
            for key in res_dict.keys():
                hist_dict[key].append(res_dict[key])
    
    # save and plot histories
    save_and_plot_histories(file_id='gaussian_white', histories=histories,
                            title='Change in accuracy with Gaussian white noise',
                            xlabel='standard deviation (\u03c3)',
                            ylabel='accuracy (\u0025)')
    
    return


# tests for gaussian color noise
def test_gaussian_color(x, y, models):
    # initialize histories
    histories = init_histories({'sigma': sigmavals})
    
    # run tests
    for sigma in sigmavals:
        # apply gaussian color noise
        print('[INFO] Applying Gaussian color noise with mu=0 and sigma={}'\
              .format(sigma))
        noisy = []
        count = 0
        for image in x:
            noisy.append(imdegrade(image, 'gaussian_color', mu=0, sigma=sigma,
                                   seed=RANDOM_SEED))
            count += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(count*100/len(x)), end='')
        
        # save noisy image samples if required
        if SAVE_NOISY:
            save_samples(noisy, SAMP_NOISY, randomize=True, seed=RANDOM_SEED,
                         filename_prefix='gaussian_color_sigma_{}'.format(sigma),
                         target_directory=OUTPUT_DIR_NOISY)
        
        # test models
        results = test_models(noisy, y, models)
        
        # update histories
        for hist_dict, res_dict in zip(histories, results):
            for key in res_dict.keys():
                hist_dict[key].append(res_dict[key])
    
    # save and plot histories
    save_and_plot_histories(file_id='gaussian_color', histories=histories,
                            title='Change in accuracy with Gaussian color noise',
                            xlabel='standard deviation (\u03c3)',
                            ylabel='accuracy (\u0025)')
    
    return


# tests for salt and pepper noise
def test_salt_and_pepper(x, y, models):
    # initialize histories
    histories = init_histories({'density': densities})
    
    # run tests
    for density in densities:
        # apply salt and pepper noise
        print('[INFO] Applying salt and pepper noise with density={}'\
              .format(density))
        noisy = []
        count = 0
        for image in x:
            noisy.append(imdegrade(image, 'salt_and_pepper', density=density,
                                   seed=RANDOM_SEED))
            count += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(count*100/len(x)), end='')
        
        # save noisy image samples if required
        if SAVE_NOISY:
            save_samples(noisy, SAMP_NOISY, randomize=True, seed=RANDOM_SEED,
                         filename_prefix='salt_and_pepper_density_{}'.format(density),
                         target_directory=OUTPUT_DIR_NOISY)
        
        # test models
        results = test_models(noisy, y, models)
        
        # update histories
        for hist_dict, res_dict in zip(histories, results):
            for key in res_dict.keys():
                hist_dict[key].append(res_dict[key])
    
    # save and plot histories
    save_and_plot_histories(file_id='salt_and_pepper', histories=histories,
                            title='Change in accuracy with salt and pepper noise',
                            xlabel='noise density',
                            ylabel='accuracy (\u0025)')
    
    return


# tests for motion blur
def test_motion_blur(x, y, models):
    # initialize histories
    histories = init_histories({'kernel_size': mb_ksizes})
    
    # run tests
    for ksize in mb_ksizes:
        # apply motion blur
        print('[INFO] Applying motion blur with kernel size=({}, {})'\
              .format(ksize, ksize))
        noisy = []
        count = 0
        mb_kernel = numpy.zeros((ksize, ksize))
        mb_kernel[ksize//2, :] = 1
        mb_kernel /= numpy.sum(mb_kernel)
        for image in x:
            noisy.append(imdegrade(image, 'motion_blur', mb_kernel=mb_kernel,
                                   seed=RANDOM_SEED))
            count += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(count*100/len(x)), end='')
        
        # save noisy image samples if required
        if SAVE_NOISY:
            save_samples(noisy, SAMP_NOISY, randomize=True, seed=RANDOM_SEED,
                         filename_prefix='motion_blur_ksize_({}x{})'.format(ksize, ksize),
                         target_directory=OUTPUT_DIR_NOISY)
        
        # test models
        results = test_models(noisy, y, models)
        
        # update histories
        for hist_dict, res_dict in zip(histories, results):
            for key in res_dict.keys():
                hist_dict[key].append(res_dict[key])
    
    # save and plot histories
    save_and_plot_histories(file_id='motion_blur', histories=histories,
                            title='Change in accuracy with motion blur',
                            xlabel='kernel size',
                            ylabel='accuracy (\u0025)')
    
    return


# tests for gaussian blur
def test_gaussian_blur(x, y, models):
    # initialize histories
    histories = init_histories({'kernel_size': gb_ksizes})
    
    # run tests
    for ksize in gb_ksizes:
        # apply gaussian blur
        print('[INFO] Applying Gaussian blur with kernel size=({}, {})'\
              .format(ksize, ksize))
        noisy = []
        count = 0
        for image in x:
            noisy.append(imdegrade(image, 'gaussian_blur',
                                   gb_ksize=(ksize, ksize), seed=RANDOM_SEED))
            count += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(count*100/len(x)), end='')
        
        # save noisy image samples if required
        if SAVE_NOISY:
            save_samples(noisy, SAMP_NOISY, randomize=True, seed=RANDOM_SEED,
                         filename_prefix='gaussian_blur_ksize_({}x{})'.format(ksize, ksize),
                         target_directory=OUTPUT_DIR_NOISY)
        
        # test models
        results = test_models(noisy, y, models)
        
        # update histories
        for hist_dict, res_dict in zip(histories, results):
            for key in res_dict.keys():
                hist_dict[key].append(res_dict[key])
    
    # save and plot histories
    save_and_plot_histories(file_id='gaussian_blur', histories=histories,
                            title='Change in accuracy with Gaussian blur',
                            xlabel='kernel size',
                            ylabel='accuracy (\u0025)')
    
    return


# tests for jpeg quality
def test_jpeg_quality(x, y, models):
    # initialize histories
    histories = init_histories({'image_quality': qualities})
    
    # run tests
    for quality in qualities:
        # apply jpeg compression
        print('[INFO] Applying JPEG compression with quality={}'\
              .format(quality))
        noisy = []
        count = 0
        for image in x:
            noisy.append(imdegrade(image, 'jpeg_compression', quality=quality,
                                   seed=RANDOM_SEED))
            count += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(count*100/len(x)), end='')
        
        # save noisy image samples if required
        if SAVE_NOISY:
            save_samples(noisy, SAMP_NOISY, randomize=True, seed=RANDOM_SEED,
                         filename_prefix='jpeg_quality_{}'.format(quality),
                         target_directory=OUTPUT_DIR_NOISY)
        
        # test models
        results = test_models(noisy, y, models)
        
        # update histories
        for hist_dict, res_dict in zip(histories, results):
            for key in res_dict.keys():
                hist_dict[key].append(res_dict[key])
    
    # save and plot histories
    save_and_plot_histories(file_id='jpeg_quality', histories=histories,
                            title='Change in accuracy with JPEG quality',
                            xlabel='image quality',
                            ylabel='accuracy (\u0025)',
                            invert_xaxis=True)
    
    return


# test
def test():
    # validate paths
    if not validate_paths():
        return
    
    # load data
    print('[INFO] Loading data... ', end='')
    (x, y) = load_data()
    print('done')
    
    # load models
    print('[INFO] Loading models... ', end='')
    models = load_models()
    print('done')
    
    # acknowledgement
    TEL_CLIENT._send_message(data={'chat_id': TELCHAT_ID,
                                   'text': ack_msg_beg.format(len(x)),
                                   'parse_mode': 'Markdown'})
    
    print('-'*34 + ' BEGIN TEST ' + '-'*34)
    
    # run tests
    for noise in NOISE_LIST:
        if noise.lower() == 'gaussian_white':
            test_gaussian_white(x, y, models)
        elif noise.lower() == 'gaussian_color':
            test_gaussian_color(x, y, models)
        elif noise.lower() == 'salt_and_pepper':
            test_salt_and_pepper(x, y, models)
        elif noise.lower() == 'motion_blur':
            test_motion_blur(x, y, models)
        elif noise.lower() == 'gaussian_blur':
            test_gaussian_blur(x, y, models)
        elif noise.lower() == 'jpeg_quality':
            test_jpeg_quality(x, y, models)
    
    print('-'*35 + ' END TEST ' + '-'*35)
    
    # acknowledgement
    TEL_CLIENT._send_message(data={'chat_id': TELCHAT_ID,
                                   'text': ack_msg_end,
                                   'parse_mode': 'Markdown'})
    
    return


# main
if __name__ == '__main__':
    try:
        test()
        if F_SHUTDOWN:
            shutdown()
    except:
        error = sys.exc_info()[0].__name__ if sys.exc_info()[0] is not None else 'Unknown'
        print('\n[INFO] Process interrupted [Reasons: {}]'.format(error))
        
        # acknowledgement
        TEL_CLIENT._send_message(data={'chat_id': TELCHAT_ID,
                                       'text': ack_msg_int.format(error),
                                       'parse_mode': 'Markdown'})
