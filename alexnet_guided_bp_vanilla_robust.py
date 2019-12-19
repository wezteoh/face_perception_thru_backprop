import numpy as np
import tensorflow as tf
import os

from scipy.io import savemat
from scipy.io import loadmat
from scipy.misc import imread
from scipy.misc import imsave

from alexnet_face_classifier import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class backprop_graph:
    def __init__(self, num_classes, nhid, cnn):
        self.num_classes = num_classes
        self.inputs = tf.placeholder(tf.float32, shape = [None, 227, 227, 3], name='input')
        self.labels_1hot = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.cnn = cnn(self.inputs, None, self.num_classes)
        self.cnn.preprocess()
        self.cnn.convlayers() 
        self.cnn.fc_layers(transfer_learning=False, nhid=nhid) 
    
    def classifier_graph(self, temp=3.0):
        self.probabilities = tf.nn.softmax(self.cnn.fc2/temp)
        self.probability = tf.tensordot(self.probabilities, self.labels_1hot, axes=[[1],[1]])
        self.log_probability = tf.log(self.probability)
    
    def guided_backprop_graph(self):
        self.grad_fc2 = tf.nn.relu(tf.gradients(self.probability, self.cnn.fc2)[0])
        self.grad_fc1 = tf.nn.relu(tf.gradients(self.cnn.fc2, self.cnn.fc1, grad_ys=self.grad_fc2)[0])
        self.grad_conv5 = tf.nn.relu(tf.gradients(self.cnn.fc1, self.cnn.conv5, grad_ys=self.grad_fc1)[0])
        self.grad_conv4 = tf.nn.relu(tf.gradients(self.cnn.conv5, self.cnn.conv4, grad_ys=self.grad_conv5)[0])
        self.grad_conv3 = tf.nn.relu(tf.gradients(self.cnn.conv4, self.cnn.conv3, grad_ys=self.grad_conv4)[0])
        self.grad_conv2 = tf.nn.relu(tf.gradients(self.cnn.conv3, self.cnn.conv2, grad_ys=self.grad_conv3)[0])
        self.grad_conv1 = tf.nn.relu(tf.gradients(self.cnn.conv2, self.cnn.conv1, grad_ys=self.grad_conv2)[0])
        self.grad_image = tf.nn.relu(tf.gradients(self.cnn.conv1, self.inputs, grad_ys=self.grad_conv1)[0])
        

###

def guided_backprop(graph, image, one_hot, sess):
    image = np.expand_dims(image, 0)
    one_hot = np.expand_dims(one_hot, 0)    
    saliency_map = sess.run(graph.grad_image, feed_dict={graph.inputs:image, graph.labels_1hot:one_hot})[0]
    scaling_adjustment = 1E-20
    saliency_map = np.sum(saliency_map, axis =-1)
    threshold = np.percentile(saliency_map,90)+scaling_adjustment
    saliency_map_scaled = saliency_map/threshold
    saliency_map_scaled = saliency_map_scaled* (saliency_map_scaled <= 1) +  (saliency_map_scaled >1)
    return saliency_map_scaled

    
    
    
    
    
    
    
    
    



