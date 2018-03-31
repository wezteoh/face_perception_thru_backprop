import numpy as np
import tensorflow as tf
import os

from scipy.io import savemat
from scipy.io import loadmat

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import _pickle as cPickle

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups,3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


class alexnet_face_classifier:
    def __init__(self, imgs=None, feature_maps=None, num_classes=17, keep_prob=1.0):
    
        # Parse input arguments into class variables
        self.imgs = imgs
        self.feature_maps = feature_maps
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.conv_parameters = []
        self.fc_parameters = []
        
    def preprocess(self):
        with tf.name_scope('recentering') as scope:
            mean = tf.reduce_mean(self.imgs, (-1,-2,-3), keep_dims=True)
            self.recentered_imgs = self.imgs-mean
            self.reversed_imgs = tf.reverse(self.recentered_imgs,[-1])

    def convlayers(self, training=True):
        # conv1
        k_h = 11; k_w = 11; k_d = 3; c_o = 96; s_h = 4; s_w = 4
        with tf.name_scope('conv1') as scope:
            self.conv1W = tf.Variable(tf.truncated_normal([k_h, k_w, k_d, c_o], dtype=tf.float32, stddev=1e-1),\
                                name='weights', trainable=training)
            self.conv1b = tf.Variable(tf.constant(0.0, shape=[c_o], dtype=tf.float32),\
                                 trainable=training, name='biases')
            conv1_in = conv(self.reversed_imgs, self.conv1W, self.conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
            self.conv1 = tf.nn.relu(conv1_in)
            self.conv_parameters += [self.conv1W, self.conv1b]
            
        #lrn1
        #lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn1 = tf.nn.local_response_normalization(self.conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
                                                      
        #maxpool1
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        self.maxpool1 = tf.nn.max_pool(self.lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
                                                    
        # conv2
        k_h = 5; k_w = 5; k_d = 48 ; c_o = 256; s_h = 1; s_w = 1; group = 2
        with tf.name_scope('conv2') as scope:
            self.conv2W = tf.Variable(tf.truncated_normal([k_h, k_w, k_d, c_o], dtype=tf.float32, stddev=1e-1),\
                                name='weights', trainable=training)
            self.conv2b = tf.Variable(tf.constant(0.0, shape=[c_o], dtype=tf.float32),\
                                 trainable=training, name='biases')
            conv2_in = conv(self.maxpool1, self.conv2W, self.conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            self.conv2 = tf.nn.relu(conv2_in)
            self.conv_parameters += [self.conv2W, self.conv2b]
        
        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn2 = tf.nn.local_response_normalization(self.conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

        #maxpool2
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        self.maxpool2 = tf.nn.max_pool(self.lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        #conv3
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; k_d = 256; c_o = 384; s_h = 1; s_w = 1; group = 1
        with tf.name_scope('conv3') as scope:
            self.conv3W = tf.Variable(tf.truncated_normal([k_h, k_w, k_d, c_o], dtype=tf.float32, stddev=1e-1),\
                                name='weights', trainable=training)
            self.conv3b = tf.Variable(tf.constant(0.0, shape=[c_o], dtype=tf.float32),\
                                 trainable=training, name='biases')
            conv3_in = conv(self.maxpool2, self.conv3W, self.conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            self.conv3 = tf.nn.relu(conv3_in)
            self.conv_parameters += [self.conv3W, self.conv3b]
        
        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; k_d = 192; c_o = 384; s_h = 1; s_w = 1; group = 2
        with tf.name_scope('conv4') as scope:
            self.conv4W = tf.Variable(tf.truncated_normal([k_h, k_w, k_d, c_o], dtype=tf.float32, stddev=1e-1),\
                                name='weights', trainable=training)
            self.conv4b = tf.Variable(tf.constant(0.0, shape=[c_o], dtype=tf.float32),\
                                 trainable=training, name='biases')
            conv4_in = conv(self.conv3, self.conv4W, self.conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            self.conv4 = tf.nn.relu(conv4_in)
            self.conv_parameters += [self.conv4W, self.conv4b]
    
        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; k_d=192; c_o = 256; s_h = 1; s_w = 1; group = 2
        with tf.name_scope('conv5') as scope:
            self.conv5W = tf.Variable(tf.truncated_normal([k_h, k_w, k_d, c_o], dtype=tf.float32, stddev=1e-1),\
                                name='weights', trainable=training)
            self.conv5b = tf.Variable(tf.constant(0.0, shape=[c_o], dtype=tf.float32),\
                                 trainable=training, name='biases')
            conv5_in = conv(self.conv4, self.conv5W, self.conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            self.conv5 = tf.nn.relu(conv5_in)
            self.conv_parameters += [self.conv5W, self.conv5b]
        
        shape = int(np.prod(self.conv5.get_shape()[1:]))
        self.conv5_flat = tf.reshape(self.conv5, [-1, shape])
            
            
    def fc_layers(self, training=True, transfer_learning=False, nhid=50):
        if transfer_learning == True:
            self.f_maps = self.feature_maps
        else:
            self.f_maps = self.conv5_flat
    
        #fc1
        with tf.name_scope('fc1') as scope:
            
            self.fc1W = tf.Variable(tf.random_normal([int(self.f_maps.get_shape()[1]), nhid],\
                                                         dtype=tf.float32,\
                                                         stddev=1e-2), name='weights', trainable=training)
            self.fc1b = tf.Variable(tf.random_normal(shape=[nhid], dtype=tf.float32),
                                 trainable=training, name='biases')
            fc1l = tf.nn.bias_add(tf.matmul(self.f_maps, self.fc1W), self.fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.fc_parameters += [self.fc1W, self.fc1b]
        
        #dropout
        self.fc1_drop = tf.nn.dropout(self.fc1, self.keep_prob)
        
        #fc2
        with tf.name_scope('fc2') as scope:
            self.fc2W = tf.Variable(tf.random_normal([nhid, self.num_classes],\
                                                         dtype=tf.float32,\
                                                         stddev=1e-2), name='weights', trainable=training)
            self.fc2b = tf.Variable(tf.random_normal(shape=[self.num_classes], dtype=tf.float32),
                                 trainable=training, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1_drop, self.fc2W), self.fc2b)
            #self.fc2 = tf.nn.relu(fc2l)
            self.fc2 = fc2l
            self.fc_parameters += [self.fc2W, self.fc2b]
        
    def load_weights(self, weight_file, sess, conv_only=False, fc_only=False):
        # Separated into two parts, conv layers and fc layers
        # to accomodate both transfer learning and end-to-end learning
        weights = cPickle.load(open(weight_file, "rb"))
        keys = sorted(weights.keys())
        if conv_only:
            for i in range(len(self.conv_parameters)):
                print (i, keys[i], np.shape(weights[keys[i]]))
                sess.run(self.conv_parameters[i].assign(weights[keys[i]]))
            
        elif fc_only:
            for j in range(len(self.fc_parameters)):
                print (j, keys[j], np.shape(weights[keys[j]]))
                sess.run(self.fc_parameters[j].assign(weights[keys[j]]))
                
        else:
            for i in range(len(self.conv_parameters)):
                print (i, keys[i], np.shape(weights[keys[i]]))
                sess.run(self.conv_parameters[i].assign(weights[keys[i]]))
                
            for i in range(len(self.fc_parameters)):
                j = i+len(self.conv_parameters)
                print (j, keys[j], np.shape(weights[keys[j]]))
                sess.run(self.fc_parameters[i].assign(weights[keys[j]]))
    
