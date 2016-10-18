from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import proj_constants
from PIL import Image, ImageDraw, ImageFont

# SUFFIX = "_normal_4_8_4_16k_4c"
TFRECORDS_TRAIN_DIR = os.path.join(proj_constants.DATA_DIR, 'tfrecords', 'train')
TFRECORDS_TEST_DIR = os.path.join(proj_constants.DATA_DIR, 'tfrecords', 'test')
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-4
MIN_ACCURACY = 0.96

SUMMARIES_DIR = ""
TRAIN_SUMMARY_DIR = ""
TEST_SUMMARY_DIR = ""
SAVE_DIR = ""
SAVE_PATH = ""

test_images = None
test_labels = None
test_label_vectors = None


def load_test_data():
    print("Loading test data...")
    global test_images
    global test_labels
    global test_label_vectors
    test_images, test_labels = read_all(TFRECORDS_TEST_DIR)
    test_label_vectors = proj_constants.to_label_vectors(test_labels)


def init_paths(suffix):
    global SUMMARIES_DIR
    SUMMARIES_DIR = os.path.join(proj_constants.DATA_DIR, 'summary', suffix)

    global TRAIN_SUMMARY_DIR
    TRAIN_SUMMARY_DIR = os.path.join(SUMMARIES_DIR, 'train')
    if not os.path.exists(TRAIN_SUMMARY_DIR):
        os.makedirs(TRAIN_SUMMARY_DIR)

    global TEST_SUMMARY_DIR
    TEST_SUMMARY_DIR = os.path.join(SUMMARIES_DIR, 'test')
    if not os.path.exists(TEST_SUMMARY_DIR):
        os.makedirs(TEST_SUMMARY_DIR)

    global SAVE_DIR
    SAVE_DIR = os.path.join(proj_constants.DATA_DIR, "models", suffix)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    global SAVE_PATH
    SAVE_PATH = os.path.join(SAVE_DIR, "model.ckpt")


def get_filepaths(dir):
    """Gets filepaths for all files within given directory"""
    walk = os.walk(dir)
    filepaths = []
    for x in walk:
        filenames = x[2]
        for filename in filenames:
            filepath = os.path.join(dir, filename)
            filepaths.append(filepath)
    return filepaths


def read_single_example(filename_queue):
    """Returns an op which reads single example from given queue"""
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
    label = tf.cast(features['y'], tf.int64)
    return image, label


def read_batches(dir, batch_size):
    """Returns an op which can be used to get batches of examples"""
    filepaths = get_filepaths(dir)
    filename_queue = tf.train.string_input_producer(filepaths, num_epochs=EPOCHS)
    image, label = read_single_example(filename_queue)
    image_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size=batch_size, num_threads=4, capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
    return image_batch, label_batch


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


def gamma_variable(shape, name):
    initial = tf.constant(1., shape=shape, name=name)
    return tf.Variable(initial)


def beta_variable(shape, name):
    initial = tf.constant(0., shape=shape, name=name)
    return tf.Variable(initial)


def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_2x2(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1]):
    return tf.nn.avg_pool(x, ksize=ksize, strides=strides, padding='VALID')


def local_response_normalization(x):
    return tf.nn.local_response_normalization(x)


def batch_normalization(x, mean, var, beta, gamma):
    return tf.nn.batch_normalization(x, mean, var, offset=beta, scale=gamma, variance_epsilon=0.001)


def read_all(dir):
    """Reads all the files present in the directory. This can be useful for testing a bunch of data at once."""
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
    """Shows first image from given arguments and prints out label. Can be useful for sanity checks."""
    image = images_eval[0]
    label = labels_eval[0]
    image = image * 255
    image = np.reshape(image, (proj_constants.WIDTH, proj_constants.HEIGHT))
    im = Image.fromarray(image)
    im.show()
    print("label id = %d" % np.argmax(label))
    print("label = %s" % proj_constants.get_label_name(np.argmax(label)))


def show_all():
    images, labels = read_all(TFRECORDS_TRAIN_DIR)

    labels_eval = proj_constants.to_label_vectors(labels)
    show_image(images, labels_eval)


class Normal_4_8_4_16k_4c():
    def build(self):
        reduction_factor = 1
        # Input images and labels
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, shape=[None, proj_constants.WIDTH * proj_constants.HEIGHT])
            y_ = tf.placeholder(tf.int32, shape=[None, proj_constants.CLASSES])
            x_image = tf.reshape(x, [-1, proj_constants.WIDTH, proj_constants.HEIGHT, 1])

        with tf.name_scope("conv_1"):
            layer1_maps = 4
            W_conv1 = weight_variable([5, 5, 1, layer1_maps], name="weight_conv_1")
            b_conv1 = bias_variable([layer1_maps], name="bias_conv_1")
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_norm1 = local_response_normalization(h_conv1)
            h_pool1 = max_pool_2x2(h_norm1)
            output_1 = h_pool1
            reduction_factor *= 2

        with tf.name_scope("conv_2"):
            layer2_maps = 8
            W_conv2 = weight_variable([5, 5, layer1_maps, layer2_maps], name="weight_conv_2")
            b_conv2 = bias_variable([layer2_maps], name="bias_conv_2")
            conv2 = conv2d(output_1, W_conv2) + b_conv2
            mean2, var2 = tf.nn.moments(conv2, [0, 1, 2])
            beta2 = beta_variable([layer2_maps], name="beta_2")
            gamma2 = gamma_variable([layer2_maps], name="gamma_2")
            norm2 = batch_normalization(conv2, mean2, var2, beta2, gamma2)
            relu2 = tf.nn.relu(norm2)
            output_2 = relu2

        with tf.name_scope("conv_3"):
            layer3_maps = 4
            W_conv3 = weight_variable([5, 5, layer2_maps, layer3_maps], name="weight_conv_3")
            b_conv3 = bias_variable([layer3_maps], name="bias_conv_3")
            conv3 = conv2d(output_2, W_conv3) + b_conv3
            mean3, var3 = tf.nn.moments(conv3, [0, 1, 2])
            beta3 = beta_variable([layer3_maps], name="beta_3")
            gamma3 = gamma_variable([layer3_maps], name="gamma_3")
            norm3 = batch_normalization(conv3, mean3, var3, beta3, gamma3)
            output_3 = norm3

        with tf.name_scope("sum"):
            sum1 = output_3 # don't sum with output_1
            relu_sum1 = tf.nn.relu(sum1)
            #sum_1 = output_3
            output_sum = relu_sum1

        with tf.name_scope("conv_4"):
            layer4_maps = 16
            W_conv4 = weight_variable([5, 5, layer3_maps, layer4_maps], name="weight_conv_4")
            b_conv4 = bias_variable([layer4_maps], name="bias_conv_4")
            h_conv4 = tf.nn.relu(conv2d(output_sum, W_conv4) + b_conv4)
            h_norm4 = local_response_normalization(h_conv4)
            h_pool4 = max_pool_2x2(h_norm4)
            reduction_factor *= 2
            h_pool4_flat = tf.reshape(h_pool4, [-1, int(proj_constants.WIDTH / reduction_factor) *
                                            int(proj_constants.HEIGHT / reduction_factor) * layer4_maps])

        with tf.name_scope("fully_connected_1"):
            fc1_size = 1024
            W_fc1 = weight_variable([int(proj_constants.WIDTH / reduction_factor)
                                     * int(proj_constants.HEIGHT / reduction_factor) * layer4_maps, fc1_size],
                                    name="weight_fully_connected_1")
            b_fc1 = bias_variable([fc1_size], name="bias_fully_connected_1")
            h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope("softmax_1"):
            W_fc2 = weight_variable([fc1_size, proj_constants.CLASSES], "weight_softmax_1")
            b_fc2 = bias_variable([proj_constants.CLASSES], "bias_softmax_1")
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.cast(y_, tf.float32) * tf.log(y_conv), reduction_indices=[1]))
            tf.scalar_summary("Cross Entropy", cross_entropy)

        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary("accuracy", accuracy)

        return "normal_4_8_4_16k_4c", x, y_, keep_prob, train_step, accuracy

class Normal_8_8_8_16k_4c():
    def build(self):
        reduction_factor = 1
        # Input images and labels
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, shape=[None, proj_constants.WIDTH * proj_constants.HEIGHT])
            y_ = tf.placeholder(tf.int32, shape=[None, proj_constants.CLASSES])
            x_image = tf.reshape(x, [-1, proj_constants.WIDTH, proj_constants.HEIGHT, 1])

        with tf.name_scope("conv_1"):
            layer1_maps = 8
            W_conv1 = weight_variable([5, 5, 1, layer1_maps], name="weight_conv_1")
            b_conv1 = bias_variable([layer1_maps], name="bias_conv_1")
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_norm1 = local_response_normalization(h_conv1)
            h_pool1 = max_pool_2x2(h_norm1)
            output_1 = h_pool1
            reduction_factor *= 2

        with tf.name_scope("conv_2"):
            layer2_maps = 8
            W_conv2 = weight_variable([5, 5, layer1_maps, layer2_maps], name="weight_conv_2")
            b_conv2 = bias_variable([layer2_maps], name="bias_conv_2")
            conv2 = conv2d(output_1, W_conv2) + b_conv2
            norm2 = local_response_normalization(conv2)
            relu2 = tf.nn.relu(norm2)
            output_2 = relu2

        with tf.name_scope("conv_3"):
            layer3_maps = 8
            W_conv3 = weight_variable([5, 5, layer2_maps, layer3_maps], name="weight_conv_3")
            b_conv3 = bias_variable([layer3_maps], name="bias_conv_3")
            conv3 = conv2d(output_2, W_conv3) + b_conv3
            norm3 = local_response_normalization(conv3)
            output_3 = norm3

        with tf.name_scope("sum"):
            sum1 = output_3 # don't sum with output_1
            relu_sum1 = tf.nn.relu(sum1)
            #sum_1 = output_3
            output_sum = relu_sum1

        with tf.name_scope("conv_4"):
            layer4_maps = 16
            W_conv4 = weight_variable([5, 5, layer3_maps, layer4_maps], name="weight_conv_4")
            b_conv4 = bias_variable([layer4_maps], name="bias_conv_4")
            h_conv4 = tf.nn.relu(conv2d(output_sum, W_conv4) + b_conv4)
            h_norm4 = local_response_normalization(h_conv4)
            h_pool4 = max_pool_2x2(h_norm4)
            reduction_factor *= 2
            h_pool4_flat = tf.reshape(h_pool4, [-1, int(proj_constants.WIDTH / reduction_factor) *
                                            int(proj_constants.HEIGHT / reduction_factor) * layer4_maps])

        with tf.name_scope("fully_connected_1"):
            fc1_size = 1024
            W_fc1 = weight_variable([int(proj_constants.WIDTH / reduction_factor)
                                     * int(proj_constants.HEIGHT / reduction_factor) * layer4_maps, fc1_size],
                                    name="weight_fully_connected_1")
            b_fc1 = bias_variable([fc1_size], name="bias_fully_connected_1")
            h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope("softmax_1"):
            W_fc2 = weight_variable([fc1_size, proj_constants.CLASSES], "weight_softmax_1")
            b_fc2 = bias_variable([proj_constants.CLASSES], "bias_softmax_1")
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.cast(y_, tf.float32) * tf.log(y_conv), reduction_indices=[1]))
            tf.scalar_summary("Cross Entropy", cross_entropy)

        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary("accuracy", accuracy)

        return "normal_8_8_8_16k_4c", x, y_, keep_prob, train_step, accuracy


class Normal_8_8_8_8_8_16k_6c():
    def build(self):
        reduction_factor = 1
        # Input images and labels
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, shape=[None, proj_constants.WIDTH * proj_constants.HEIGHT])
            y_ = tf.placeholder(tf.int32, shape=[None, proj_constants.CLASSES])
            x_image = tf.reshape(x, [-1, proj_constants.WIDTH, proj_constants.HEIGHT, 1])

        with tf.name_scope("conv_1"):
            layer1_maps = 8
            W_conv1 = weight_variable([5, 5, 1, layer1_maps], name="weight_conv_1")
            b_conv1 = bias_variable([layer1_maps], name="bias_conv_1")
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_norm1 = local_response_normalization(h_conv1)
            h_pool1 = max_pool_2x2(h_norm1)
            output_1 = h_pool1
            reduction_factor *= 2

        with tf.name_scope("conv_2"):
            layer2_maps = 8
            W_conv2 = weight_variable([5, 5, layer1_maps, layer2_maps], name="weight_conv_2")
            b_conv2 = bias_variable([layer2_maps], name="bias_conv_2")
            conv2 = conv2d(output_1, W_conv2) + b_conv2
            relu2 = tf.nn.relu(conv2)
            norm2 = local_response_normalization(relu2)
            output_2 = norm2

        with tf.name_scope("conv_3"):
            layer3_maps = 8
            W_conv3 = weight_variable([5, 5, layer2_maps, layer3_maps], name="weight_conv_3")
            b_conv3 = bias_variable([layer3_maps], name="bias_conv_3")
            conv3 = conv2d(output_2, W_conv3) + b_conv3
            relu3 = tf.nn.relu(conv3)
            norm3 = local_response_normalization(relu3)
            output_3 = norm3

        with tf.name_scope("conv_4"):
            layer4_maps = 8
            W_conv4 = weight_variable([5, 5, layer3_maps, layer4_maps], name="weight_conv_4")
            b_conv4 = bias_variable([layer4_maps], name="bias_conv_4")
            conv4 = conv2d(output_3, W_conv4) + b_conv4
            relu4 = tf.nn.relu(conv4)
            norm4 = local_response_normalization(relu4)
            output_4 = norm4

        with tf.name_scope("conv_5"):
            layer5_maps = 8
            W_conv5 = weight_variable([5, 5, layer4_maps, layer5_maps], name="weight_conv_5")
            b_conv5 = bias_variable([layer5_maps], name="bias_conv_5")
            conv5 = conv2d(output_4, W_conv5) + b_conv5
            relu5 = tf.nn.relu(conv5)
            norm5 = local_response_normalization(relu5)
            output_5 = norm5

        with tf.name_scope("conv_6"):
            layer6_maps = 16
            W_conv6 = weight_variable([5, 5, layer5_maps, layer6_maps], name="weight_conv_6")
            b_conv6 = bias_variable([layer6_maps], name="bias_conv_6")
            conv6 = conv2d(output_5, W_conv6) + b_conv6
            relu6 = tf.nn.relu(conv6)
            h_norm6 = local_response_normalization(relu6)
            h_pool6 = max_pool_2x2(h_norm6)
            reduction_factor *= 2
            h_pool6_flat = tf.reshape(h_pool6, [-1, int(proj_constants.WIDTH / reduction_factor) *
                                            int(proj_constants.HEIGHT / reduction_factor) * layer6_maps])

        with tf.name_scope("fully_connected_1"):
            fc1_size = 1024
            W_fc1 = weight_variable([int(proj_constants.WIDTH / reduction_factor)
                                     * int(proj_constants.HEIGHT / reduction_factor) * layer6_maps, fc1_size],
                                    name="weight_fully_connected_1")
            b_fc1 = bias_variable([fc1_size], name="bias_fully_connected_1")
            h_fc1 = tf.nn.relu(tf.matmul(h_pool6_flat, W_fc1) + b_fc1)
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope("softmax_1"):
            W_fc2 = weight_variable([fc1_size, proj_constants.CLASSES], "weight_softmax_1")
            b_fc2 = bias_variable([proj_constants.CLASSES], "bias_softmax_1")
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.cast(y_, tf.float32) * tf.log(y_conv), reduction_indices=[1]))
            tf.scalar_summary("Cross Entropy", cross_entropy)

        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary("accuracy", accuracy)

        return "Normal_8_8_8_8_8_16k_6c", x, y_, keep_prob, train_step, accuracy





class Normal_8_8_16_16_16_32k_6c_with_pool():
    def build(self):
        reduction_factor = 1
        # Input images and labels
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, shape=[None, proj_constants.WIDTH * proj_constants.HEIGHT])
            y_ = tf.placeholder(tf.int32, shape=[None, proj_constants.CLASSES])
            x_image = tf.reshape(x, [-1, proj_constants.WIDTH, proj_constants.HEIGHT, 1])

        with tf.name_scope("conv_1"):
            layer1_maps = 8
            W_conv1 = weight_variable([5, 5, 1, layer1_maps], name="weight_conv_1")
            b_conv1 = bias_variable([layer1_maps], name="bias_conv_1")
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_norm1 = local_response_normalization(h_conv1)
            h_pool1 = max_pool_2x2(h_norm1)
            output_1 = h_pool1
            reduction_factor *= 2

        with tf.name_scope("conv_2"):
            layer2_maps = 8
            W_conv2 = weight_variable([5, 5, layer1_maps, layer2_maps], name="weight_conv_2")
            b_conv2 = bias_variable([layer2_maps], name="bias_conv_2")
            conv2 = conv2d(output_1, W_conv2) + b_conv2
            relu2 = tf.nn.relu(conv2)
            norm2 = local_response_normalization(relu2)
            pool2 = max_pool_2x2(norm2)
            reduction_factor *= 2
            output_2 = pool2

        with tf.name_scope("conv_3"):
            layer3_maps = 16
            W_conv3 = weight_variable([5, 5, layer2_maps, layer3_maps], name="weight_conv_3")
            b_conv3 = bias_variable([layer3_maps], name="bias_conv_3")
            conv3 = conv2d(output_2, W_conv3) + b_conv3
            relu3 = tf.nn.relu(conv3)
            norm3 = local_response_normalization(relu3)
            pool3 = max_pool_2x2(norm3)
            reduction_factor *= 2
            output_3 = pool3

        with tf.name_scope("conv_4"):
            layer4_maps = 16
            W_conv4 = weight_variable([5, 5, layer3_maps, layer4_maps], name="weight_conv_4")
            b_conv4 = bias_variable([layer4_maps], name="bias_conv_4")
            conv4 = conv2d(output_3, W_conv4) + b_conv4
            relu4 = tf.nn.relu(conv4)
            norm4 = local_response_normalization(relu4)
            pool4 = max_pool_2x2(norm4)
            reduction_factor *= 2
            output_4 = pool4

        with tf.name_scope("conv_5"):
            layer5_maps = 16
            W_conv5 = weight_variable([5, 5, layer4_maps, layer5_maps], name="weight_conv_5")
            b_conv5 = bias_variable([layer5_maps], name="bias_conv_5")
            conv5 = conv2d(output_4, W_conv5) + b_conv5
            relu5 = tf.nn.relu(conv5)
            norm5 = local_response_normalization(relu5)
            output_5 = norm5

        with tf.name_scope("conv_6"):
            layer6_maps = 32
            W_conv6 = weight_variable([5, 5, layer5_maps, layer6_maps], name="weight_conv_6")
            b_conv6 = bias_variable([layer6_maps], name="bias_conv_6")
            conv6 = conv2d(output_5, W_conv6) + b_conv6
            relu6 = tf.nn.relu(conv6)
            h_norm6 = local_response_normalization(relu6)
            h_pool6 = max_pool_2x2(h_norm6)
            reduction_factor *= 2
            h_pool6_flat = tf.reshape(h_pool6, [-1, int(proj_constants.WIDTH / reduction_factor) *
                                            int(proj_constants.HEIGHT / reduction_factor) * layer6_maps])

        with tf.name_scope("fully_connected_1"):
            fc1_size = 1024
            W_fc1 = weight_variable([int(proj_constants.WIDTH / reduction_factor)
                                     * int(proj_constants.HEIGHT / reduction_factor) * layer6_maps, fc1_size],
                                    name="weight_fully_connected_1")
            b_fc1 = bias_variable([fc1_size], name="bias_fully_connected_1")
            h_fc1 = tf.nn.relu(tf.matmul(h_pool6_flat, W_fc1) + b_fc1)
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope("softmax_1"):
            W_fc2 = weight_variable([fc1_size, proj_constants.CLASSES], "weight_softmax_1")
            b_fc2 = bias_variable([proj_constants.CLASSES], "bias_softmax_1")
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.cast(y_, tf.float32) * tf.log(y_conv), reduction_indices=[1]))
            tf.scalar_summary("Cross Entropy", cross_entropy)

        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary("accuracy", accuracy)

        return "Normal_8_8_16_16_16_32k_6c_with_pool", x, y_, keep_prob, train_step, accuracy


class Resnet_4_8_4_16k_4c:

    def build(self):
        reduction_factor = 1
        # Input images and labels
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, shape=[None, proj_constants.WIDTH * proj_constants.HEIGHT])
            y_ = tf.placeholder(tf.int32, shape=[None, proj_constants.CLASSES])
            x_image = tf.reshape(x, [-1, proj_constants.WIDTH, proj_constants.HEIGHT, 1])

        with tf.name_scope("conv_1"):
            layer1_maps = 4
            W_conv1 = weight_variable([5, 5, 1, layer1_maps], name="weight_conv_1")
            b_conv1 = bias_variable([layer1_maps], name="bias_conv_1")
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_norm1 = local_response_normalization(h_conv1)
            h_pool1 = max_pool_2x2(h_norm1)
            output_1 = h_pool1
            reduction_factor *= 2

        with tf.name_scope("conv_2"):
            layer2_maps = 8
            W_conv2 = weight_variable([5, 5, layer1_maps, layer2_maps], name="weight_conv_2")
            b_conv2 = bias_variable([layer2_maps], name="bias_conv_2")
            conv2 = conv2d(output_1, W_conv2) + b_conv2
            mean2, var2 = tf.nn.moments(conv2, [0, 1, 2])
            beta2 = beta_variable([layer2_maps], name="beta_2")
            gamma2 = gamma_variable([layer2_maps], name="gamma_2")
            norm2 = batch_normalization(conv2, mean2, var2, beta2, gamma2)
            print("norm2")
            print(norm2)
            relu2 = tf.nn.relu(norm2)
            output_2 = relu2

        with tf.name_scope("conv_3"):
            layer3_maps = 4
            W_conv3 = weight_variable([5, 5, layer2_maps, layer3_maps], name="weight_conv_3")
            b_conv3 = bias_variable([layer3_maps], name="bias_conv_3")
            conv3 = conv2d(output_2, W_conv3) + b_conv3
            mean3, var3 = tf.nn.moments(conv3, [0, 1, 2])
            beta3 = beta_variable([layer3_maps], name="beta_3")
            gamma3 = gamma_variable([layer3_maps], name="gamma_3")
            norm3 = batch_normalization(conv3, mean3, var3, beta3, gamma3)
            print("norm3")
            print(norm3)
            output_3 = norm3

        with tf.name_scope("sum"):
            sum1 = output_1 + output_3
            relu_sum1 = tf.nn.relu(sum1)
            #sum_1 = output_3
            output_sum = relu_sum1

        with tf.name_scope("conv_4"):
            layer4_maps = 16
            W_conv4 = weight_variable([5, 5, layer3_maps, layer4_maps], name="weight_conv_4")
            b_conv4 = bias_variable([layer4_maps], name="bias_conv_4")
            h_conv4 = tf.nn.relu(conv2d(output_sum, W_conv4) + b_conv4)
            h_norm4 = local_response_normalization(h_conv4)
            h_pool4 = max_pool_2x2(h_norm4)
            reduction_factor *= 2
            h_pool4_flat = tf.reshape(h_pool4, [-1, int(proj_constants.WIDTH / reduction_factor) *
                                            int(proj_constants.HEIGHT / reduction_factor) * layer4_maps])

        with tf.name_scope("fully_connected_1"):
            fc1_size = 1024
            W_fc1 = weight_variable([int(proj_constants.WIDTH / reduction_factor)
                                     * int(proj_constants.HEIGHT / reduction_factor) * layer4_maps, fc1_size],
                                    name="weight_fully_connected_1")
            b_fc1 = bias_variable([fc1_size], name="bias_fully_connected_1")
            h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope("softmax_1"):
            W_fc2 = weight_variable([fc1_size, proj_constants.CLASSES], "weight_softmax_1")
            b_fc2 = bias_variable([proj_constants.CLASSES], "bias_softmax_1")
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.cast(y_, tf.float32) * tf.log(y_conv), reduction_indices=[1]))
            tf.scalar_summary("Cross Entropy", cross_entropy)

        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary("accuracy", accuracy)

        return "resnet_4_8_4_16k_4c", x, y_, keep_prob, train_step, accuracy


# Latest models:

class Normal_paper():
    def build(self):
        reduction_factor = 1
        # Input images and labels
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, shape=[None, proj_constants.WIDTH * proj_constants.HEIGHT])
            y_ = tf.placeholder(tf.int32, shape=[None, proj_constants.CLASSES])
            x_image = tf.reshape(x, [-1, proj_constants.WIDTH, proj_constants.HEIGHT, 1])

        with tf.name_scope("conv1"):
            maps1 = 4
            weight1 = weight_variable([5, 5, 1, maps1], name="weight1")
            bias1 = bias_variable([maps1], name="bias1")
            conv1 = conv2d(x_image, weight1) + bias1
            relu1 = tf.nn.relu(conv1)
            norm1 = local_response_normalization(relu1)
            pool1 = max_pool_2x2(norm1)
            reduction_factor *= 2
            output1 = tf.nn.relu(pool1)

        with tf.name_scope("conv2"):
            maps2 = 12
            weight2 = weight_variable([5, 5, maps1, maps2], name="weight2")
            bias2 = bias_variable([maps2], name="bias2")
            conv2 = conv2d(output1, weight2) + bias2
            relu2 = tf.nn.relu(conv2)
            norm2 = local_response_normalization(relu2)
            pool2 = max_pool_2x2(norm2)
            reduction_factor *= 2
            output2 = tf.nn.relu(pool2)

        with tf.name_scope("conv3"):
            maps3 = 16
            weight3 = weight_variable([5, 5, maps2, maps3], name="weight3")
            bias3 = bias_variable([maps3], name="bias3")
            conv3 = conv2d(output2, weight3) + bias3
            relu3 = tf.nn.relu(conv3)
            output3 = relu3
            flat = tf.reshape(output3, [-1, int(proj_constants.WIDTH / reduction_factor) *
                                        int(proj_constants.HEIGHT / reduction_factor) * maps3])

        with tf.name_scope("dropout"):
            keep_prob = tf.placeholder(tf.float32)
            dropout = tf.nn.dropout(flat, keep_prob)

        with tf.name_scope("fc1"):
            fc1_size = 1024
            weight_fc1 = weight_variable([int(proj_constants.WIDTH / reduction_factor) *
                                        int(proj_constants.HEIGHT / reduction_factor) * maps3, fc1_size],
                                         name="weight_fc1")
            bias_fc1 = bias_variable([fc1_size], name="bias_fc1")
            fc1 = tf.matmul(dropout, weight_fc1) + bias_fc1
            output_fc1 = tf.nn.relu(fc1)

        with tf.name_scope("softmax1"):
            weight_sm1 = weight_variable([fc1_size, proj_constants.CLASSES], "weight_sm1")
            bias_sm1 = bias_variable([proj_constants.CLASSES], "bias_sm1")
            y_conv = tf.nn.softmax(tf.matmul(output_fc1, weight_sm1) + bias_sm1)

        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.cast(y_, tf.float32) * tf.log(y_conv), reduction_indices=[1]))
            tf.scalar_summary("Cross Entropy", cross_entropy)

        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary("accuracy", accuracy)

        return "Normal_paper", x, y_, keep_prob, train_step, accuracy

class Normal_4_16_16_32():
    def build(self):
        reduction_factor = 1
        # Input images and labels
        kernel_size = 3
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, shape=[None, proj_constants.WIDTH * proj_constants.HEIGHT])
            y_ = tf.placeholder(tf.int32, shape=[None, proj_constants.CLASSES])
            x_image = tf.reshape(x, [-1, proj_constants.WIDTH, proj_constants.HEIGHT, 1])

        with tf.name_scope("conv1"):
            maps1 = 4
            weight1 = weight_variable([kernel_size, kernel_size, 1, maps1], name="weight1")
            bias1 = bias_variable([maps1], name="bias1")
            conv1 = conv2d(x_image, weight1) + bias1
            relu1 = tf.nn.relu(conv1)
            norm1 = local_response_normalization(relu1)
            pool1 = max_pool_2x2(norm1)
            reduction_factor *= 2
            output1 = pool1

        with tf.name_scope("conv2"):
            maps2 = 16
            weight2 = weight_variable([kernel_size, kernel_size, maps1, maps2], name="weight2")
            bias2 = bias_variable([maps2], name="bias2")
            conv2 = conv2d(output1, weight2) + bias2
            relu2 = tf.nn.relu(conv2)
            norm2 = local_response_normalization(relu2)
            pool2 = max_pool_2x2(norm2)
            reduction_factor *= 2
            output2 = pool2

        with tf.name_scope("conv3"):
            maps3 = 16
            weight3 = weight_variable([kernel_size, kernel_size, maps2, maps3], name="weight3")
            bias3 = bias_variable([maps3], name="bias3")
            conv3 = conv2d(output2, weight3) + bias3
            relu3 = tf.nn.relu(conv3)
            norm3 = local_response_normalization(relu3)
            pool3 = max_pool_2x2(norm3)
            reduction_factor *= 2
            output3 = pool3

        with tf.name_scope("conv4"):
            maps4 = 32
            weight4 = weight_variable([kernel_size, kernel_size, maps3, maps4], name="weight4")
            bias4 = bias_variable([maps4], name="bias4")
            conv4 = conv2d(output3, weight4) + bias4
            relu4 = tf.nn.relu(conv4)
            output4 = relu4
            flat = tf.reshape(output4, [-1, int(proj_constants.WIDTH / reduction_factor) *
                                        int(proj_constants.HEIGHT / reduction_factor) * maps4])

        with tf.name_scope("fc1"):
            fc1_size = 1024
            weight_fc1 = weight_variable([int(proj_constants.WIDTH / reduction_factor) *
                                        int(proj_constants.HEIGHT / reduction_factor) * maps4, fc1_size],
                                         name="weight_fc1")
            bias_fc1 = bias_variable([fc1_size], name="bias_fc1")
            fc1 = tf.matmul(flat, weight_fc1) + bias_fc1
            output_fc1 = tf.nn.relu(fc1)

        with tf.name_scope("dropout"):
            keep_prob = tf.placeholder(tf.float32)
            dropout = tf.nn.dropout(output_fc1, keep_prob)

        with tf.name_scope("softmax1"):
            weight_sm1 = weight_variable([fc1_size, proj_constants.CLASSES], "weight_sm1")
            bias_sm1 = bias_variable([proj_constants.CLASSES], "bias_sm1")
            y_conv = tf.nn.softmax(tf.matmul(dropout, weight_sm1) + bias_sm1)

        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.cast(y_, tf.float32) * tf.log(y_conv), reduction_indices=[1]))
            tf.scalar_summary("Cross Entropy", cross_entropy)

        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary("accuracy", accuracy)

        return "Normal_4_16_16_32", x, y_, keep_prob, train_step, accuracy


def train(model):

    with tf.Graph().as_default():
        model_name, x, y_, keep_prob, train_step, accuracy = model.build()
        print("Training with model: %s" %model_name)
        init_paths(model_name)
        highest_accuracy = 0
        # Create and initialize the ops
        images, labels = read_batches(TFRECORDS_TRAIN_DIR, batch_size=BATCH_SIZE)
        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())
        sess = tf.InteractiveSession()
        merge_summary = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(TRAIN_SUMMARY_DIR, sess.graph)
        test_writer = tf.train.SummaryWriter(TEST_SUMMARY_DIR)
        model_saver = tf.train.Saver()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Restore model if a checkpoint is found
        if os.path.exists(SAVE_PATH):
            print("Previous checkpoint found. Restoring the model...")
            model_saver.restore(sess, SAVE_PATH)

        step_num = 0

        print("Starting the training...")
        try:
            while not coord.should_stop():
                images_eval, labels_eval = sess.run([images, labels])
                label_vectors = proj_constants.to_label_vectors(labels_eval)
                train_step.run(feed_dict={x: images_eval, y_: label_vectors, keep_prob: 0.45})

                if step_num % 20 == 0:
                    # Evaluate train accuracy every 10th step
                    summary, train_accuracy = sess.run([merge_summary, accuracy], feed_dict={x: images_eval, y_: label_vectors, keep_prob: 1.0})
                    train_writer.add_summary(summary, step_num)
                    print("Step: %d Training accuracy: %g" % (step_num, train_accuracy))

                if step_num % 200 == 0:
                    # Evaluate test accuracy every 100th step
                    summary, test_accuracy = sess.run([merge_summary, accuracy], feed_dict={x: test_images, y_: test_label_vectors, keep_prob: 1.0})
                    test_writer.add_summary(summary, step_num)
                    print("Step: %d Test accuracy: %g" % (step_num, test_accuracy))

                    if test_accuracy > MIN_ACCURACY and test_accuracy > highest_accuracy:
                        print("Saving the model...")
                        saved_path = model_saver.save(sess, SAVE_PATH)
                        print("Saved the model in file: %s" % saved_path)
                        highest_accuracy = test_accuracy
                    train_writer.flush()
                    test_writer.flush()

                step_num += 1
        except tf.errors.OutOfRangeError:
            print("Ending training...")
            print("Total Steps = %d" % step_num)
        finally:
            coord.request_stop()
            train_writer.flush()
            test_writer.flush()
        coord.join(threads)
        sess.close()

load_test_data()
# train(Normal_4_8_4_16k_4c())
# train(Resnet_4_8_4_16k_4c())
# train(Normal_8_8_8_16k_4c())
# train(Normal_8_8_8_8_8_16k_6c())
# train(Normal_8_8_16_16_16_32k_6c_with_pool())
# train(Normal_paper())
train(Normal_4_16_16_32())