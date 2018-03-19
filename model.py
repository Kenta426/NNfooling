""" a simple network with two convolution kernels followed by a fully connected layer.
classify images by softmax
"""
import numpy as np
from params import Params
import pickle
import tensorflow as tf
from utils import *
from batch import Batch
import os
import matplotlib.pyplot as plt


class ConvNet(object):
    def __init__(self, fool = False):
        self.key2char = Params.key2char
        self.char2key = Params.char2key
        self.train_data = Params.train_data
        self.train_labels = Params.train_labels
        self.test_data = Params.test_data
        self.test_labels = Params.test_labels
        self.input = tf.placeholder(tf.float32, [None, Params.image_dim[0]*Params.image_dim[1]])
        self.output = tf.placeholder(tf.float32, [None,  len(self.char2key.keys())])
        self.trainable = (not fool)
        self.fool_trainable = fool
        self.pred = self.build_model()

    def fooling_layer(self, inputs, name = 'fooling'):
        with tf.variable_scope(name):
            ffilter = tf.get_variable('filter', shape=[Params.image_dim[0]*Params.image_dim[1]], initializer=tf.truncated_normal_initializer(0.05))
            self.ffilter = tf.scalar_mul(Params.alpha, ffilter)
            return tf.nn.bias_add(inputs, self.ffilter)

    def conv_layer(self, inputs, name = 'conv'):
        with tf.variable_scope(name):
            weight_shape = [Params.filter_shape[0], Params.filter_shape[1], Params.channel, Params.num_filter]
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
        inputs = self.input
        inputs = tf.reshape(inputs, [-1, Params.image_dim[0], Params.image_dim[1], Params.channel])

        conv = self.conv_layer(inputs, 'conv-1')
        pool = self.pooling_layer(conv, 'pool-1')

        self.final_layer = pool

        output_shape = int(Params.image_dim[1] * (0.5 ** (Params.convolution_layer)))
        reshape = tf.reshape(self.final_layer, [-1, output_shape*output_shape*Params.num_filter])
        fc = self.fully_connected(reshape, Params.fc_unit, 'fc-1')
        s_max = self.soft_max(fc, Params.fc_unit, 'soft_max')
        return s_max

    def run(self):
        with open(self.train_data, 'rb') as f:
            data = pickle.load(f)
        with open(self.train_labels, 'rb') as f:
            labels = pickle.load(f)
        labels = one_hot(sorted(list(set(labels))), labels)

        with open(self.test_data, 'rb') as f:
            data_t = pickle.load(f)
        with open(self.test_labels, 'rb') as f:
            labels_t = pickle.load(f)
        labels_t = one_hot(sorted(list(set(labels_t))), labels_t)

        b = Batch(data, labels, Params.batch_size)

        var = tf.trainable_variables()
        conv = [v for v in var if v.name.startswith("conv")]
        fool = [v for v in var if v.name.startswith("fooling")]
        fc = [v for v in var if v.name.startswith("fc")]
        smax = [v for v in var if v.name.startswith("soft_max")]


        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.output))
        optimiser = tf.train.AdamOptimizer(Params.learning_rate).minimize(cross_entropy, var_list=conv + fc + smax)
        # collect prediction in the batch
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        total_batch = int(len(data) / Params.batch_size)

        learning = []
        if self.trainable:
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(Params.epoch):
                    b.shuffle()
                    avg_cost = 0
                    print ("{} epoch".format(epoch))
                    for i in range(total_batch):
                        batch_x, batch_y = b.next_batch()
                        _, cost = sess.run([optimiser, cross_entropy], feed_dict={model.input: batch_x, model.output: batch_y})
                        avg_cost += cost/total_batch
                    acc = sess.run(accuracy, feed_dict={model.input: data_t, model.output: labels_t})
                    learning.append(acc)
                    # saving the model
                    if epoch % 10 == 0:
                        pass
                        # checkpoint_path = os.path.join(Params.checkpoint_path, 'model.ckpt')
                        # save_path = saver.save(sess, checkpoint_path)
                        # print("model saved to {}".format(checkpoint_path))
                        # print (filters)

                    # print(avg_cost, acc)
        plt.plot(learning)
        plt.title('Epoch vs Test accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Test accuracy')
        plt.show()


    def fool(self):
        with open(self.train_data, 'rb') as f:
            data = pickle.load(f)
        with open(self.train_labels, 'rb') as f:
            labels = pickle.load(f)
        key = sorted(list(set(labels)))
        labels = one_hot(key, labels)
        saver = tf.train.Saver()

        _, target = fool_target(key, labels[0])

        var = tf.trainable_variables()
        fool = [v for v in var if v.name.startswith("fool")]
        print (fool)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.output))
        optimiser = tf.train.AdamOptimizer(Params.learning_rate).minimize(cross_entropy, var_list=fool)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, "model/fool/model.ckpt")

            print("Model restored.")
            print ("true label :{}".format(np.argmax(labels[0])))
            print ("target label :{}".format(np.argmax(target)))
            initial = sess.run(model.pred, feed_dict={model.input: data[0].reshape(1,-1)})
            print (initial)
            for epoch in range(100):
                print ("{} epoch".format(epoch))
                for i in range(1000):
                    c, f, cost = sess.run([cross_entropy, model.ffilter, model.pred], feed_dict={model.input: data[0].reshape(1,-1), model.output: target.reshape(1,-1)})
                    print (f)
                print (c, cost)
                # saving the model
            plt.imshow(f.reshape(28,28).T)
            plt.show()


if __name__ == '__main__':
    tf.reset_default_graph()
    model = ConvNet()
    # Restore variables from disk.
    model.run()
