import numpy as np
import tensorflow as tf
import os

from scipy.io import savemat
from scipy.io import loadmat

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import _pickle as cPickle
from alexnet_face_classifier import *

f1, f2 ,f3 = open("training_set_17_227.pkl","rb"), open("validation_set_17_227.pkl","rb"), open("test_set_17_227.pkl","rb")
training_set, validation_set, test_set = cPickle.load(f1), cPickle.load(f2), cPickle.load(f3)

inputs = tf.placeholder(tf.float32, shape = [None, 227, 227, 3])
extractor = alexnet_face_classifier(inputs)
extractor.preprocess()
extractor.convlayers()

with tf.Session() as sess:
    extractor.load_weights('alexnet_weights.pkl', sess, True)
    for set in [training_set, validation_set, test_set]:
        for actor in set:
            set[actor] = sess.run(extractor.conv5_flat, feed_dict={inputs: set[actor]})
            print(actor + ' completed') 

cPickle.dump(training_set, open("training_set_17_conv5.pkl","wb"))
cPickle.dump(validation_set, open("validation_set_17_conv5.pkl","wb"))
cPickle.dump(test_set, open("test_set_17_conv5.pkl","wb"))
