""" a simple network with two convolution kernels followed by a fully connected layer.
classify images by softmax
"""
import numpy as np
from params import Params
import pickle
import tensorflow as tf
from utils import *

class ConvNet(object):
    def __init__(self):
        self.key2char = Params.key2char
        self.char2key = Params.char2key
        self.train_data = Params.train_data
        self.train_labels = Params.train_labels
        self.input = tf.placeholder(tf.float32, [None, Params.image_dim[0]*Params.image_dim[1]])
        self.output = tf.placeholder(tf.float32, [None,  len(self.char2key.keys())])
        self.pred = self.build_model()

    def conv_layer(self, inputs, name = 'conv'):
        with tf.variable_scope(name):
            weight_shape = [Params.image_dim[0], Params.image_dim[1], Params.channel, Params.num_filter]
            kernel = tf.get_variable('w', weight_shape, initializer = tf.truncated_normal_initializer(stddev = 0.05))
            biases = tf.get_variable('b', shape=[Params.num_filter], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            return tf.nn.relu(tf.nn.bias_add(conv, biases))

    def pooling_layer(self, inputs, name):
        with tf.variable_scope(name):
            pool = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
            return pool

    def fully_connected(self, inputs, output, name):
         with tf.variable_scope(name):
            units_in = inputs.get_shape().as_list()[-1]
            weights = tf.get_variable('w', [units_in, output], initializer = tf.truncated_normal_initializer(stddev = 0.05))
            biases = tf.get_variable('b', shape=[output], initializer=tf.constant_initializer(0.1))
            return tf.nn.relu(tf.nn.xw_plus_b(inputs, weights, biases))

    def soft_max(self, inputs, units_in, name):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [units_in, len(self.char2key.keys())], initializer = tf.truncated_normal_initializer(stddev = 0.05))
            b = tf.get_variable('b', shape=[len(self.char2key.keys())], initializer=tf.constant_initializer(0.0))
            self.logits = tf.nn.xw_plus_b(inputs, w, b)
            return tf.nn.softmax(self.logits)

    def build_model(self):
        inputs = tf.reshape(self.input, [-1, Params.image_dim[0], Params.image_dim[1], Params.channel])
        for i in range(Params.convolution_layer):
            conv = self.conv_layer(inputs, 'conv-'+str(i+1))
            pool = self.pooling_layer(conv, 'pool-'+str(i+1))
            inputs = pool

        self.final_layer = inputs
        output_shape = int(Params.image_dim[1] * (0.5 ** (Params.convolution_layer)))
        reshape = tf.reshape(self.final_layer, [-1, output_shape*output_shape*Params.num_filter])
        fc = self.fully_connected(reshape, len(self.char2key.keys()), 'fc-1')
        s_max = self.soft_max(fc, len(self.char2key.keys()), 'soft_max')
        return s_max

    def run(self):
        with open(self.train_data, 'rb') as f:
            data = pickle.load(f)
        with open(self.train_labels, 'rb') as f:
            labels = pickle.load(f)
        labels = one_hot(sorted(list(set(labels))), labels)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.output))
        optimiser = tf.train.AdamOptimizer(Params.learning_rate).minimize(cross_entropy)
        # collect prediction in the batch
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            _, acc = sess.run([optimiser, accuracy], feed_dict={model.input: data[:10], model.output: labels[:10]})
            print(acc)

if __name__ == '__main__':
    model = ConvNet()
    model.run()
    # print (len(model.char2key.keys()))
