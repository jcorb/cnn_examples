
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import time


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

## set up place holders
n_x = 784 # number of pixels
n_y = 10 # number of classes
x = tf.placeholder(tf.float32, shape=[None, n_x])
y_ = tf.placeholder(tf.float32, shape=[None, n_y])


## Convolution weights
## filter size is 5 x 5, input depth is 1 (greyscale images), number of filters is 32
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

## reshape input to be images of 28 x 28 pixels
x_image = tf.reshape(x, [-1, 28, 28, 1])

## apply the ReLU activation to the convolved arrays
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
## use max pool to downsample the output - reduces it to 14 x 14
h_pool1 = max_pool_2x2(h_conv1)

## second convolution layer
## filter size is 5 x 5, input depth is 32, number of filters is 64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

## apply the ReLU activation to the convolved arrays
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
## use max pool to downsample the output - reduces it to 7 x 7
h_pool2 = max_pool_2x2(h_conv2)


## set up connected layer weights, size is 7 x 7 x 64, number of neurons is 1024
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

##flatten hpool2
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
## compute layer activations using ReLU
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

##set up droput regularization
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## set up second connected layer (output layer) weights
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


##set up model
## use softmax_cross_entropy as the loss caluclation
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

##use an Adam Optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## run the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 500 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                  x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))






