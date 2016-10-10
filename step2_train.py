from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import proj_constants
from PIL import Image, ImageDraw, ImageFont

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
            'y': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['x'], tf.float64)
    image = tf.cast(image, tf.float32)
    image.set_shape([proj_constants.WIDTH * proj_constants.HEIGHT])

    # label = tf.decode_raw(features['y'], tf.float64)
    label = tf.cast(features['y'], tf.int64)
    # label.set_shape([proj_constants.CLASSES])
    return image, label


def inputs(dir, batch_size):
    filepaths = get_filepaths(dir)
    filename_queue = tf.train.string_input_producer(filepaths, num_epochs=EPOCHS)
    image, label = read_single_example(filename_queue)
    image_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size=batch_size, num_threads=4, capacity=500 + 3 * batch_size, min_after_dequeue=500)
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
    print("y_conv")
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.cast(y_, tf.float32) * tf.log(y_conv), reduction_indices=[1]))
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y_, keep_prob, train_step, accuracy


def get_all_records(dir):
    filepaths = get_filepaths(dir)
    filename_queue = tf.train.string_input_producer(filepaths, num_epochs=1)
    image, label = read_single_example(filename_queue)
    init_op = tf.group(tf.initialize_local_variables(), tf.initialize_all_variables())
    images = []
    labels = []
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while True:
                curr_image, curr_label = sess.run([image, label])
                images.append(curr_image)
                labels.append(curr_label)
        except tf.errors.OutOfRangeError, e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)
    return images, labels


def show_image(images_eval, labels_eval):
    image = images_eval[1]
    label = labels_eval[1]
    image = image * 255
    image = np.reshape(image, (proj_constants.WIDTH, proj_constants.HEIGHT))
    im = Image.fromarray(image)
    im.show()
    print("label id = %d" % np.argmax(label))
    print("label = %s" % proj_constants.get_label_name(np.argmax(label)))


def show_all():
    images, labels = get_all_records(TFRECORDS_TRAIN_DIR)

    labels_eval = proj_constants.to_label_vectors(labels)
    show_image(images, labels_eval)


def train_CNN():
    """Train data for a number of steps."""

    with tf.Graph().as_default():
        x, y_, keep_prob, train_step, accuracy = build_CNN()
        images, labels = inputs(TFRECORDS_TRAIN_DIR, batch_size=BATCH_SIZE)
        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())
        sess = tf.InteractiveSession()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        iter = 0
        try:
            while not coord.should_stop():
                images_eval, labels_eval = sess.run([images, labels])
                label_vectors = proj_constants.to_label_vectors(labels_eval)
                train_step.run(feed_dict={x: images_eval, y_: label_vectors, keep_prob: 0.5})
                if iter % 100 == 0:
                    print("Iteration: %d" % iter)
                    train_accuracy = accuracy.eval(feed_dict={x: images_eval, y_: label_vectors, keep_prob: 1.0})
                    print("Training accuracy %g" % train_accuracy)
                iter += 1
        except tf.errors.OutOfRangeError:
            print("Ending training...")
            print("iter = %d" % iter)
        finally:
            test_images, test_labels = get_all_records(TFRECORDS_TEST_DIR)
            test_label_vectors = proj_constants.to_label_vectors(test_labels)
            test_accuracy = accuracy.eval(feed_dict={x: test_images, y_: test_label_vectors, keep_prob: 1.0})
            print("Test accuracy %g" % test_accuracy)
            coord.request_stop()

        coord.join(threads)
        sess.close()


train_CNN()
#show_all()
# print(get_filepaths(TFRECORDS_TRAIN_DIR))
