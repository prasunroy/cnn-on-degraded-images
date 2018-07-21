# -*- coding: utf-8 -*-
"""
Building blocks for a capsule network.
Created on Tue May 29 11:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/cnn-on-degraded-images

Original author: Xifeng Guo
Original source: https://github.com/XifengGuo/CapsNet-Keras

"""


# imports
from __future__ import division

import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras import layers


# Length class
class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    
    def get_config(self):
        return super(Length, self).get_config()


# Mask class
class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = K.sqrt(K.sum(K.square(inputs), -1))
            mask = K.one_hot(indices=K.argmax(x, 1),
                             num_classes=x.get_shape().as_list()[1])
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked
    
    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])
    
    def get_config(self):
        return super(Mask, self).get_config()


# CapsuleLayer class
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform', **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        return
    
    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(shape=[self.num_capsule,
                                        self.input_num_capsule,
                                        self.dim_capsule,
                                        self.input_dim_capsule],
                                        initializer=self.kernel_initializer,
                                        name='W')
        self.built = True
        return
    
    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(inputs, 1)
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]),
                              elems=inputs_tiled)
        b = tf.zeros(shape=[K.shape(inputs_hat)[0],
                            self.num_capsule,
                            self.input_num_capsule])
        assert self.routings > 0
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))
            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        return outputs
    
    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])
    
    def get_config(self):
        config = {'num_capsule': self.num_capsule,
                  'dim_capsule': self.dim_capsule,
                  'routings': self.routings}
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# squash
def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm
                                                           + K.epsilon())
    return scale * vectors


# PrimaryCaps
def PrimaryCaps(inputs, dim_capsule, n_channels, kernel_size,
                strides, padding):
    output = layers.Conv2D(filters=dim_capsule*n_channels,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           name='PrimaryCaps_Conv2D')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule],
                             name='PrimaryCaps_Reshape')(output)
    return layers.Lambda(squash, name='PrimaryCaps_Squash')(outputs)
