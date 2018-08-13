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
    for i in range(2001): ### was 20000
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


    # compute in batches to avoid OOM on GPUs 
    accuracy_l = []
    for _ in range(20):
      batch = mnist.test.next_batch(500, shuffle=False)
      accuracy_l.append(accuracy.eval(feed_dict={x: batch[0], 
                                                 y_: batch[1], 
                                                 keep_prob: 1.0}))
    test_accu = numpy.mean(accuracy_l)
    print('test accuracy %g' % test_accu)


    # Find the list of all images of 2 from mnist.test
    # The list has length 1032, which is not too large. So keep it on the go.
    print('Finding images of 2...')
    twos_list = []
    for i in range(len(mnist.test.labels)):
      if mnist.test.labels[i]==2:
        twos_list.append(i)

    # Define function to modify images
    def modify_img(eps, img, label, epochs, plot=False):
      '''
      Modify img by stepping up the y=6 mountain multiple times.
      -----
      eps: float, epsilon so that in each step new_image = -epsilon*gradient + old_image
      img: numpy.ndarray, original img from mnist.test.images
      label: int, original label of img from mnist.test.labels
      epochs: int, number of steps we take up the hill
      plot: bool, set True to return valures required for graph plotting
      -----
      Returns: 
      delta: list, difference between old image and new image (-epsilon*gradient)
      new_pred: list of one int, classification of new image by the model
      new_probs[0][6]: float, the probability of the new image being a 6, given by the model; confidence of the model
      new_x: numpy.ndarray, new image
      probs[0][pred[0]]: float, confidence of model in classifying the original image
      pred[0]: int, in range [0,9], prediction of the original image by model
      '''
      new_x = img.reshape([-1,784])
      original_y = label.reshape([-1])
      if plot:
        probs, pred = sess.run([tf.nn.softmax(y_conv), prediction], feed_dict={x:new_x, y_:original_y, keep_prob:1})
      for i in range(epochs):
        # Find gradient on the y=6 mountain
        gradient6 = sess.run([grad_x], feed_dict={x:new_x, y_:[6], keep_prob:1})
        gradient6 = numpy.reshape(gradient6,(1,784))
        # Modify new image
        delta = -eps*numpy.sign(gradient6)
        new_x = delta + new_x
        # Truncate values so the values in new image is in [0,1]
        new_x[new_x>1] = 1
        new_x[new_x<0] = 0

      # Run model on new image, find new prediction and the probability the model assigns to each of 0-9.
      if not plot:
        new_pred = sess.run(prediction, feed_dict={x: new_x, y_:original_y, keep_prob:1})
        return new_pred
      else:
        new_probs = sess.run(tf.nn.softmax(y_conv), feed_dict={x: new_x, y_:original_y, keep_prob:1})
        return delta, new_probs[0][6], new_x, probs[0][pred[0]], pred[0]
    
    # An auxiliary function to plot figures
    def plot_in_fig(fig, image, fig_no, rows, cols):
      fig.add_subplot(rows, cols, fig_no)
      plt.imshow(image.reshape(28,28))
      plt.xticks([])
      plt.yticks([])
      plt.gray()
    
    def plot_image(eps, epochs, rows=10, cols=3):
      '''
      Select 10 images randomly to plot the corresponding graphs. Save the figure in the end.
      -----
      Input:
      eps: float, epsilon so that in each step new_image = -epsilon*gradient + old_image
      epochs: int, number of steps we take up the hill
      rows=10: we select 10 images randomly from MNIST.test
      cols=3: corresponds to the columns of original image, delta, modified image
      -----
      Return: void
      '''
      to_print = numpy.random.choice(twos_list, 10)
      fig = plt.figure(figsize=(8,26))
      fig_no = 1
      counter = 0
      for i in to_print:
        img = mnist.test.images[i]
        label = mnist.test.labels[i]
        delta, new_prob, new_x, prob, pred = modify_img(eps, img, label, epochs, plot=True)
        
        plot_in_fig(fig, img, fig_no, rows, cols)
        plt.title("%d \n %.2f confidence" % (pred, 100*prob))

        plot_in_fig(fig, delta, fig_no+1, rows, cols)
        plt.title("delta")

        plot_in_fig(fig, new_x, fig_no+2, rows, cols)
        plt.title("6\n %.2f confidence" % (100*new_prob))

        fig_no += 3

      plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                  wspace=None, hspace=2)
      plt.show()
      fig.savefig('eps=%.2f_epochs=%i_test_accu%.2f.png' %(eps,epochs, test_accu))

    
    def modify_all_images_and_plot(eps=0.1, epochs=8):
      '''
      Apply modify_img() to see the ratio of images of 2 which we've successfully turned into a 6. Then plot graphs.
      Shown above each graph in first column (original graphs) is the predicted label, and the corresponding confidence in model.
      Shown in the second column are the modifications, i.e. difference between modified and original graphs.
      Shown above each graph in third column (modifed graphs) is the probability that the graph is a 6, given by the model.
      Note: We did not scale delta to be in range [0,1]. But we can see its noisiness from the plotted graphs (although they are not noises).
      -----
      Input:
      eps: float, epsilon so that in each step new_image = -epsilon*gradient + old_image
      epochs: int, number of steps we take up the hill
      -----
      Return: void
      '''
      print('Modifying images...')
      counter = 0
      for i in twos_list:
        img = mnist.test.images[i]
        label = mnist.test.labels[i]
        pred_after_mod = modify_img(eps, img, label, epochs)
        if pred_after_mod == [6]:
          counter += 1
      print('Ratio of 6 after stepping up on y=6: %.2f' % (counter/len(twos_list)))
      plot_image(eps, epochs)
    
    


    # Run the functions
    modify_all_images_and_plot(eps=0.1, epochs=8)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)