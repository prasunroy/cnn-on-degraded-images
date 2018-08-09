# -*- coding: utf-8 -*-
"""
Degradation models.
Created on Thu May 24 11:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/cnn-on-degraded-images

"""


# imports
import cv2
import numpy
import random


# apply a degradation model on an image
def imdegrade(image, model, mu=0, sigma=0, density=0, gb_ksize=(1, 1),
              mb_kernel=numpy.zeros((1, 1), dtype='uint8'), quality=100,
              seed=None):
    # setup seeds for random number generators
    # (only required for reproducibility)
    numpy.random.seed(seed)
    random.seed(seed)
    
    # create a copy of the input image to prevent direct modification
    # on the original input image
    image = image.copy()
    
    # add an extra dimension for color channel
    # (only required for grayscale images)
    if len(image.shape) == 2:
        image = numpy.expand_dims(image, 2)
    
    # get dimension of the image
    h, w, c = image.shape
    
    # apply a degradation model
    model = model.lower()
    
    if model == 'gaussian_white' and sigma > 0:
        image = image / 255.0
        noise = numpy.random.normal(mu, sigma, (h, w))
        noise = numpy.dstack([noise]*c)
        image += noise
        image = numpy.clip(image, 0, 1)
        image = (image * 255.0).astype('uint8')
    
    elif model == 'gaussian_color' and sigma > 0:
        image = image / 255.0
        noise = numpy.random.normal(mu, sigma, (h, w, c))
        image += noise
        image = numpy.clip(image, 0, 1)
        image = (image * 255.0).astype('uint8')
    
    elif model == 'salt_and_pepper':
        if density < 0:
            density = 0
        elif density > 1:
            density = 1
        x = random.sample(range(w), w)
        y = random.sample(range(h), h)
        x, y = numpy.meshgrid(x, y)
        xy = numpy.c_[x.reshape(-1), y.reshape(-1)]
        n = int(w * h * density)
        n = random.sample(range(w*h), n)
        for i in n:
            if random.random() > 0.5:
                image[xy[i][1], xy[i][0], :] = 255
            else:
                image[xy[i][1], xy[i][0], :] = 0
    
    elif model == 'motion_blur':
        image = cv2.filter2D(image, -1, mb_kernel,
                             borderType=cv2.BORDER_CONSTANT)
    
    elif model == 'gaussian_blur':
        image = cv2.GaussianBlur(image, gb_ksize, 0,
                                 borderType=cv2.BORDER_CONSTANT)
    
    elif model == 'jpeg_compression':
        if quality < 0:
            quality = 0
        elif quality > 100:
            quality = 100
        image = cv2.imencode('.jpg', image,
                             [int(cv2.IMWRITE_JPEG_QUALITY), quality])[-1]
        image = cv2.imdecode(image, -1)
    
    # remove the extra dimension for color channel
    # (only required for grayscale images)
    if image.shape[-1] == 1:
        image = numpy.squeeze(image, 2)
    
    return image
