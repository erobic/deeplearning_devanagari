from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as mpplot
import os
import matplotlib.image as mpimg
import proj_constants

# Constants used for dealing with the files, matches convert_to_records.
TFRECORDS_TRAIN_DIR = os.path.join(proj_constants.DATA_DIR, 'tfrecords', 'train')
TFRECORDS_TEST_DIR = os.path.join(proj_constants.DATA_DIR, 'tfrecords', 'test')
BATCH_SIZE = 220
EPOCHS = 300
LEARNING_RATE = 1e-4


def get_filepaths(dir):
    walk = os.walk(dir)
    filepaths = []
    for x in walk:
        filenames = x[2]
        for filename in filenames:
            filepath = os.path.join(dir, filename)
            filepaths.append(filepath)
    return filepaths


def read_single_example(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'x': tf.FixedLenFeature([], tf.string),
            'y': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['x'], tf.float64)
    image = tf.cast(image, tf.float32)
    image.set_shape([proj_constants.WIDTH * proj_constants.HEIGHT])

    label = tf.decode_raw(features['y'], tf.float64)
    label = tf.cast(label, tf.int32)
    label.set_shape([proj_constants.CLASSES])
    return image, label


def inputs(dir, batch_size):
    """Reads input data num_epochs times.

    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.

    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, image_size]
        in the range [0, 1].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, label_size).
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """
    with tf.name_scope('input'):
        filepaths = get_filepaths(dir)
        filename_queue = tf.train.string_input_producer(filepaths, num_epochs=EPOCHS)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_single_example(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=500 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=500)
        return image_batch, label_batch


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def build_CNN():
    # Input images and labels.
    x = tf.placeholder(tf.float32, shape=[None, proj_constants.WIDTH * proj_constants.HEIGHT])
    y_ = tf.placeholder(tf.int32, shape=[None, proj_constants.CLASSES])

    # Build a CNN
    print("Configuring CNN...")
    x_image = tf.reshape(x, [-1, proj_constants.WIDTH, proj_constants.HEIGHT, 1])

    # 1st layer
    # Filter
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    print("h_conv1")
    print(h_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    print("h_pool1")
    print(h_pool1)

    # 2nd layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    print("h_conv2")
    print(h_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    print("h_pool2")
    print(h_pool2)
    # image size would have reduced by a factor of 4. Of course we have to account for all the channels too
    h_pool2_flat = tf.reshape(h_pool2, [-1, int(proj_constants.WIDTH / 4) * int(proj_constants.HEIGHT / 4)
                                        * 64])
    print("h_pool2_flat")
    print(h_pool2_flat)

    # 3rd layer
    W_fc1 = weight_variable([int(proj_constants.WIDTH / 4) * int(proj_constants.HEIGHT / 4)
                             * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, proj_constants.CLASSES])
    b_fc2 = bias_variable([proj_constants.CLASSES])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    print("Defining cost function...")
    # Define cost function
    print("y_")
    print(y_)
    print("y_conv")
    print(y_conv)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.cast(y_, tf.float32) * tf.log(y_conv), reduction_indices=[1]))
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y_, keep_prob, train_step, accuracy


def show_image(images_eval):
    image = images_eval[20]
    image = image * 255
    image = np.reshape(image, (proj_constants.WIDTH, proj_constants.HEIGHT))
    mpplot.imshow(image, cmap='gray')
    mpplot.show()


def get_all_records(FILE):
    filename_queue = tf.train.string_input_producer([FILE], num_epochs=1, shuffle=True)
    image, label = read_single_example(filename_queue)
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())
    sess = tf.InteractiveSession()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    images = []
    labels = []
    try:
        while True:
            images.append(image.eval())
            labels.append(label.eval())
    except tf.errors.OutOfRangeError, e:
        coord.request_stop(e)

    finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()
    return images, labels


def train_CNN():
    """Train data for a number of steps."""

    with tf.Graph().as_default():
        x, y_, keep_prob, train_step, accuracy = build_CNN()
        images, labels = inputs(TFRECORDS_TRAIN_DIR, batch_size=BATCH_SIZE)
        #test_images, test_labels = inputs(TEST_FILE, batch_size=BATCH_SIZE)

        print("Starting session...")
        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())

        sess = tf.InteractiveSession()
        print("Initializing all variables...")

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        iter = 0
        print("Starting the training of CNN...")
        try:
            while not coord.should_stop():
                images_eval = images.eval()
                labels_eval = labels.eval()
                print(labels_eval)
                for lbl in labels_eval:
                    print(np.argmax(lbl))
                train_step.run(feed_dict={x: images_eval, y_: labels_eval, keep_prob: 0.5})
                if iter % 100 == 0:
                    print("Iteration: %d" % iter)
                    train_accuracy = accuracy.eval(feed_dict={x: images_eval, y_: labels_eval, keep_prob: 1.0})
                    print("Training accuracy %g" % train_accuracy)
                iter += 1
        except tf.errors.OutOfRangeError:
            print("Ending training...")
            print("iter = %d" % iter)
        finally:
            test_images, test_labels = get_all_records(TFRECORDS_TEST_DIR)
            test_accuracy = accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})
            print("Test accuracy %g" % test_accuracy)
            coord.request_stop()

        coord.join(threads)
        sess.close()

train_CNN()
#print(get_filepaths(TFRECORDS_TRAIN_DIR))
