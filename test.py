# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import numpy

### MY MODIFICATIONS 1 BEGIN
# import plt to plot
import matplotlib.pyplot as plt
### MY MODIFICATIONS 1 END

FLAGS = None

### MY MODIFICATIONS 1 BEGIN
# Pass y_ into deepnn(), in order to define 'loss' inside deepnn()
def deepnn(x, y_):
### MY MODIFICATIONS 1 END
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  ### MY MODIFICATIONS 1 BEGIN
  # Move difinitions of loss here, from function main(_)
  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)
  # define grad_x, the derivative of loss (cross_entropy) with respect to input x
  grad_x = tf.gradients(ys=cross_entropy, xs=x)

  # In addition to y_conv, return also grad_x and cross_entropy. 
  return y_conv, grad_x, cross_entropy, keep_prob
  ### MY MODIFICATIONS 1 END


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.int64, [None])

  ### MY MODIFICATIONS 1 BEGIN
  # Build the graph for the deep net (minor modifications due to change to input of deepnn)
  y_conv, grad_x, cross_entropy, keep_prob = deepnn(x, y_)
  ### MY MODIFICATIONS 1 END

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    ### MY MODIFICATIONS 1 BEGIN
    # define prediction = tf.argmax(y_conv, 1), to be used later
    prediction = tf.argmax(y_conv, 1)
    ### END OF MY 1 MODIFICATIONS
    correct_prediction = tf.equal(prediction, y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000): ### was 20000
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # Find the list of all images of 2 from mnist.test
    twos_list = []
    for i in range(len(mnist.test.labels)):
      if mnist.test.labels[i]==2:
        twos_list.append(i)

    # Define a function to modify images in two ways
    def modify_img(eps, img, label):
      # Calculate gradient = d(loss)/dx, where loss is the cross entropy when y=2 (true label)
      gradient = sess.run([grad_x], feed_dict={x:img.reshape([-1,784]), y_:label.reshape([-1]), keep_prob:1})
      gradient = numpy.reshape(gradient, (1,784))
      # Calculate gradient6 = d(loss6)/dx, where loss6 is the cross entropy when y=6 (target label)
      gradient6 = sess.run([grad_x], feed_dict={x:img.reshape([-1,784]), y_:[6], keep_prob:1})
      gradient6 = numpy.reshape(gradient6,(1,784))

      # First: modify image by adding to it 0.1*sign(d(loss)/dx) 
      # and subtracting from it 0.1*sign(d(loss6)/dx)
      # so we step down on the y=6 mountain, and step up on the y=2 mountain.
      new_img2to6 = eps*numpy.sign(gradient-gradient6) + img.reshape([-1,784])
      new_img2to6 = new_img2to6.reshape([-1,784])
      new_pred2to6 = sess.run([prediction], feed_dict={x: new_img2to6, y_:label.reshape([-1]), keep_prob:1})

      # Second: modify image by subtracting from it 0.1*sign(d(loss6)/dx)
      # so we step down on the y=6 mountain only.
      new_img6 = -eps*numpy.sign(gradient6) + img.reshape([-1,784])
      new_img6 = new_img6.reshape([-1,784])
      new_pred6 = sess.run([prediction], feed_dict={x:new_img6, y_:label.reshape([-1]), keep_prob:1})

      # Also calculate the prediction if we only step down the y=2 mountain (as in the previous version)
      new_img = eps*numpy.sign(gradient) + img.reshape([-1,784])
      new_img = new_img.reshape([-1,784])
      new_pred = sess.run([prediction], feed_dict={x: new_img, y_:label.reshape([-1]), keep_prob:1})
      return new_pred, new_pred2to6, new_pred6
    
    # Run modify_img() to see how well each of the two methods work.
    counter = 0
    counter2to6 = 0
    counter6 = 0
    for i in twos_list:
      img = mnist.test.images[i]
      label = mnist.test.labels[i]
      pred_after_mod, pred_after_mod2to6, pred_after_mod6 = modify_img(0.25, img, label)
      if pred_after_mod == [6]:
        counter += 1
      if pred_after_mod2to6 == [6]:
        counter2to6 += 1
      if pred_after_mod6 == [6]:
        counter6 += 1
    print("Ratio of 6 after stepping down on y=2: %.2f" % (counter/len(twos_list)))
    print('Ratio of 6 after stepping down on y=2 and up on y=6: %.2f' % (counter2to6/len(twos_list)))
    print('Ratio of 6 after stepping up on y=6: %.2f' % (counter6/len(twos_list)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)