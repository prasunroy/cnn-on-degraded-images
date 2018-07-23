# -*- coding: utf-8 -*-
"""
Pipeline utilities.
Created on Sun Jul 22 22:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/cnn-on-degraded-images

"""


# imports
import cv2
import os
import platform
import random


# save samples
def save_samples(images=[], n=0, randomize=False, seed=None,
                 filename_prefix='', target_directory='.'):
    
    # setup seeds for random number generators
    # (only required for reproducibility)
    random.seed(seed)
    
    # required if n > total number of images and/or n < 0
    n = max(min(len(images), n), 0)
    
    # select subset of images by list indices
    indices = [i for i in range(n) if not randomize] + \
              [i for i in random.sample(range(len(images)), n) if randomize]
    
    # save images
    for i, j in enumerate(indices):
        filepath = os.path.join(target_directory,
                                filename_prefix+'_{}.png'.format(i))
        cv2.imwrite(filepath, images[j])
    
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
