from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import proj_constants
from PIL import Image, ImageDraw, ImageFont

TFRECORDS_TRAIN_DIR = os.path.join(proj_constants.DATA_DIR, 'tfrecords', 'train')
TFRECORDS_TEST_DIR = os.path.join(proj_constants.DATA_DIR, 'tfrecords', 'test')
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 8e-4
SUMMARIES_DIR = os.path.join(proj_constants.DATA_DIR, 'summary')
TRAIN_SUMMARY_DIR = os.path.join(SUMMARIES_DIR, 'train')
TEST_SUMMARY_DIR = os.path.join(SUMMARIES_DIR, 'test')
MIN_ACCURACY = 0.97
SAVE_PATH = os.path.join(proj_constants.DATA_DIR, "model.ckpt")


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


def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_2x2(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1]):
    return tf.nn.avg_pool(x, ksize=ksize, strides=strides, padding='VALID')


def local_response_normaliztion(x):
    return tf.nn.local_response_normalization(x)


def build_CNN():
    """Builds a conv net to train the model

    Returns:
        x: placeholder for input images
        y_: placeholder for actual labels
        keep_prob: dropout parameter (1 = no dropout)
        train_step: Optimizer to train the model
        accuracy: op to calculate accuracy of data
    """
    # Input images and labels
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, proj_constants.WIDTH * proj_constants.HEIGHT])
        y_ = tf.placeholder(tf.int32, shape=[None, proj_constants.CLASSES])
        x_image = tf.reshape(x, [-1, proj_constants.WIDTH, proj_constants.HEIGHT, 1])

    with tf.name_scope("conv_1"):
        layer1_maps = 16
        W_conv1 = weight_variable([5, 5, 1, layer1_maps], name="weight_conv_1")
        b_conv1 = bias_variable([layer1_maps], name="bias_conv_1")
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_norm1 = local_response_normaliztion(h_conv1)
        h_pool1 = max_pool_2x2(h_norm1)

    with tf.name_scope("conv_2"):
        layer2_maps = 32
        W_conv2 = weight_variable([5, 5, layer1_maps, layer2_maps], name="weight_conv_2")
        b_conv2 = bias_variable([layer2_maps], name="bias_conv_2")
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_norm2 = local_response_normaliztion(h_conv2)
        h_pool2 = max_pool_2x2(h_norm2)

    with tf.name_scope("conv_3"):
        layer3_maps = 32
        W_conv3 = weight_variable([5, 5, layer2_maps, layer3_maps], name="weight_conv_3")
        b_conv3 = bias_variable([layer3_maps], name="bias_conv_3")
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_norm3 = local_response_normaliztion(h_conv3)
        h_pool3 = max_pool_2x2(h_norm3)

    with tf.name_scope("conv_4"):
        layer4_maps = 64
        W_conv4 = weight_variable([5, 5, layer3_maps, layer4_maps], name="weight_conv_4")
        b_conv4 = bias_variable([layer4_maps], name="bias_conv_4")
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_norm4 = local_response_normaliztion(h_conv4)
        h_pool4 = max_pool_2x2(h_norm4)
        image_reduction_factor = 16
        h_pool4_flat = tf.reshape(h_pool4, [-1, int(proj_constants.WIDTH / image_reduction_factor) *
                                        int(proj_constants.HEIGHT / image_reduction_factor) * layer4_maps])

    with tf.name_scope("fully_connected_1"):
        fc1_size = 1024
        W_fc1 = weight_variable([int(proj_constants.WIDTH / image_reduction_factor)
                                 * int(proj_constants.HEIGHT / image_reduction_factor) * layer4_maps, fc1_size],
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

    return x, y_, keep_prob, train_step, accuracy


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


def build_ResNet():
    """Builds a ResNet to train the model

    Returns:
        x: placeholder for input images
        y_: placeholder for actual labels
        keep_prob: dropout parameter (1 = no dropout)
        train_step: Optimizer to train the model
        accuracy: op to calculate accuracy of data
    """
    image_reduction_factor = 1
    # Input images and labels
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, proj_constants.WIDTH * proj_constants.HEIGHT])
        y_ = tf.placeholder(tf.int32, shape=[None, proj_constants.CLASSES])
        keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(x, [-1, proj_constants.WIDTH, proj_constants.HEIGHT, 1])

    with tf.name_scope("preprocess"):
        with tf.name_scope("preprocess_1"):
            p1_maps = 16
            W_p1 = weight_variable([3, 3, 1, p1_maps], name="weight_p1")
            b_p1 = bias_variable([p1_maps], name="bias_p1")
            conv_p1 = conv2d(x_image, W_p1) + b_p1
            relu_p1 = tf.nn.relu(conv_p1)
            norm_p1 = local_response_normaliztion(relu_p1)
            output_p1 = norm_p1
            print("relu_p1")
            print(relu_p1)
        with tf.name_scope("preprocess_2"):
            p2_maps = 16
            W_p2 = weight_variable([3, 3, p1_maps, p2_maps], name="weight_p2")
            b_p2 = bias_variable([p2_maps], name="bias_p2")
            conv_p2 = conv2d(output_p1, W_p2) + b_p2
            relu_p2 = tf.nn.relu(conv_p2)
            norm_p2 = local_response_normaliztion(relu_p2)
            output_p2 = norm_p2
            print("relu_p2")
            print(relu_p2)
        with tf.name_scope("preprocess_3"):
            # Use convolutional layer with stride = 2 instead of max pooling layer
            p3_maps = 32
            W_p3 = weight_variable([3, 3, p2_maps, p3_maps], name="weight_p3")
            b_p3 = bias_variable([p3_maps], name="bias_p3")
            conv_p3 = conv2d(output_p2, W_p3, strides=[1, 2, 2, 1]) + b_p3
            relu_p3 = tf.nn.relu(conv_p3)
            norm_p3 = local_response_normaliztion(relu_p3)
            output_p3 = norm_p3
            image_reduction_factor *= 2
            print("relu_p3")
            print(relu_p3)
            # relu_p3 = tf.nn.relu(norm_p3)

    with tf.name_scope("res_net_1"):
        with tf.name_scope("res_net_1_conv_1"):
            r1c1_maps = 64
            W_r1c1 = weight_variable([3, 3, p3_maps, r1c1_maps], name="weight_r1c1")
            b_r1c1 = bias_variable([r1c1_maps], name="bias_r1c1")
            conv_r1c1 = conv2d(output_p3, W_r1c1) + b_r1c1
            relu_r1c1 = tf.nn.relu(conv_r1c1)
            #norm_r1c1 = local_response_normaliztion(relu_r1c1)
            output_r1c1 = relu_r1c1
            print("relu_r1c1")
            print(relu_r1c1)
        with tf.name_scope("res_net_1_conv_2"):
            r1c2_maps = 32
            W_r1c2 = weight_variable([3, 3, r1c1_maps, r1c2_maps], name="weight_r1c2")
            b_r1c2 = bias_variable([r1c2_maps], name="bias_r1c2")
            conv_r1c2 = conv2d(output_r1c1, W_r1c2) + b_r1c2
            #norm_r1c2 = local_response_normaliztion(conv_r1c2)
            output_r1c2 = conv_r1c2
            print("output_r1c2")
            print(output_r1c2)
        with tf.name_scope("res_net_1_sum"):
            sum_r1 = output_p3 + output_r1c2
            relu_sum_r1 = tf.nn.relu(sum_r1)
            output_r1 = relu_sum_r1
            print("relu_sum_r1")
            print(relu_sum_r1)
        with tf.name_scope("res_net_1_conv_3"):
            r1c3_maps = 64
            W_r1c3 = weight_variable([3, 3, r1c2_maps, r1c3_maps], name="weight_r1c3")
            b_r1c3 = bias_variable([r1c3_maps], name="bias_r1c3")
            conv_r1c3 = conv2d(output_r1, W_r1c3, strides=[1, 2, 2, 1]) + b_r1c3
            relu_r1c3 = tf.nn.relu(conv_r1c3)
            norm_r1c3 = local_response_normaliztion(relu_r1c3)
            output_r1c3 = norm_r1c3
            image_reduction_factor *= 2
            print("relu_r1c3")
            print(relu_r1c3)

    with tf.name_scope("last_layer"):
        with tf.name_scope("last_conv_1"):
            lc1_maps = proj_constants.CLASSES
            W_lc1 = weight_variable([3, 3, r1c3_maps, lc1_maps], name="weight_lc1")
            b_lc1 = bias_variable([lc1_maps], name="bias_lc1")
            conv_lc1 = conv2d(output_r1c3, W_lc1) + b_lc1
            relu_lc1 = tf.nn.relu(conv_lc1)
            norm_lc1 = local_response_normaliztion(conv_lc1)
            output_lc1 = norm_lc1
            print("relu_lc1")
            print(relu_lc1)
        with tf.name_scope("flatten1"):
            flatten1_size = int(proj_constants.WIDTH * proj_constants.HEIGHT * lc1_maps/
                                                       (image_reduction_factor * image_reduction_factor))
            flatten1 = tf.reshape(output_lc1, [-1, flatten1_size])
            print("flatten1")
            print(flatten1)
        with tf.name_scope("fully_connected1"):
            fc1_size = 1024
            W_fc1 = weight_variable([flatten1_size, fc1_size], name="weight_fc1")
            b_fc1 = bias_variable([fc1_size], name="bias_fc1")
            linear_fc1 = tf.matmul(flatten1, W_fc1) + b_fc1
            relu_fc1 = tf.nn.relu(linear_fc1)
            drop_fc1 = tf.nn.dropout(relu_fc1, keep_prob)
            output_fc1 = drop_fc1

        # with tf.name_scope("avg_pool"):
        #     avg_pool = avg_pool_2x2(output_lc1, ksize=[1, int(proj_constants.WIDTH/image_reduction_factor),
        #                                          int(proj_constants.HEIGHT/image_reduction_factor), 1])
        #     print("avg_pool")
        #     print(avg_pool)
        # with tf.name_scope("flatten"):
        #     flatten = tf.reshape(avg_pool, [-1, proj_constants.CLASSES])
        #     print("flatten")
        #     print(flatten)
        with tf.name_scope("fully_connected2"):
            fc2_size = proj_constants.CLASSES
            W_fc2 = weight_variable([fc1_size, fc2_size], name="weight_fc2")
            b_fc2 = bias_variable([fc2_size], name="bias_fc2")
            linear_fc2 = tf.matmul(output_fc1, W_fc2) + b_fc2
            output_fc2 = linear_fc2
        with tf.name_scope("soft_max"):
            soft_max = tf.nn.softmax(output_fc2)
            prediction = soft_max
            print("prediction")
            print(prediction)

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.cast(y_, tf.float32) * tf.log(prediction), reduction_indices=[1]))
        tf.scalar_summary("Cross Entropy", cross_entropy)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))

    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary("accuracy", accuracy)

    return x, y_, keep_prob, train_step, accuracy


def build_CNN_with_ResNet():
    """Builds a conv net to train the model

    Returns:
        x: placeholder for input images
        y_: placeholder for actual labels
        keep_prob: dropout parameter (1 = no dropout)
        train_step: Optimizer to train the model
        accuracy: op to calculate accuracy of data
    """
    reduction_factor = 1
    # Input images and labels
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, proj_constants.WIDTH * proj_constants.HEIGHT])
        y_ = tf.placeholder(tf.int32, shape=[None, proj_constants.CLASSES])
        x_image = tf.reshape(x, [-1, proj_constants.WIDTH, proj_constants.HEIGHT, 1])

    with tf.name_scope("conv_1"):
        layer1_maps = 32
        W_conv1 = weight_variable([5, 5, 1, layer1_maps], name="weight_conv_1")
        b_conv1 = bias_variable([layer1_maps], name="bias_conv_1")
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_norm1 = local_response_normaliztion(h_conv1)
        h_pool1 = max_pool_2x2(h_norm1)
        output_1 = h_pool1
        reduction_factor *= 2

    with tf.name_scope("conv_2"):
        layer2_maps = 32
        W_conv2 = weight_variable([5, 5, layer1_maps, layer2_maps], name="weight_conv_2")
        b_conv2 = bias_variable([layer2_maps], name="bias_conv_2")
        h_conv2 = tf.nn.relu(conv2d(output_1, W_conv2) + b_conv2)
        h_norm2 = local_response_normaliztion(h_conv2)
        output_2 = h_norm2
        #h_pool2 = max_pool_2x2(h_norm2)
        #reduction_factor *= 2

    with tf.name_scope("conv_3"):
        layer3_maps = 32
        W_conv3 = weight_variable([5, 5, layer2_maps, layer3_maps], name="weight_conv_3")
        b_conv3 = bias_variable([layer3_maps], name="bias_conv_3")
        h_conv3 = tf.nn.relu(conv2d(output_2, W_conv3) + b_conv3)
        h_norm3 = local_response_normaliztion(h_conv3)
        output_3 = h_norm3
        #h_pool3 = max_pool_2x2(h_norm3)
        #reduction_factor *= 2

    with tf.name_scope("sum"):
        sum_1 = output_1 + output_3

    with tf.name_scope("conv_4"):
        layer4_maps = 64
        W_conv4 = weight_variable([5, 5, layer3_maps, layer4_maps], name="weight_conv_4")
        b_conv4 = bias_variable([layer4_maps], name="bias_conv_4")
        h_conv4 = tf.nn.relu(conv2d(sum_1, W_conv4) + b_conv4)
        h_norm4 = local_response_normaliztion(h_conv4)
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

    return x, y_, keep_prob, train_step, accuracy



def train_CNN():
    highest_accuracy = 0
    """Trains CNN. Prints out accuracy every 100 steps. Prints out test accuracy at the end."""
    with tf.Graph().as_default():
        # Build the CNN
        #x, y_, keep_prob, train_step, accuracy = build_CNN()
        #x, y_, keep_prob, train_step, accuracy = build_ResNet()
        x, y_, keep_prob, train_step, accuracy = build_CNN_with_ResNet()

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

        # print("Loading test data...")
        # test_images, test_labels = read_all(TFRECORDS_TEST_DIR)
        # test_label_vectors = proj_constants.to_label_vectors(test_labels)
        step_num = 0

        print("Starting the training...")
        try:
            while not coord.should_stop():
                images_eval, labels_eval = sess.run([images, labels])
                label_vectors = proj_constants.to_label_vectors(labels_eval)
                train_step.run(feed_dict={x: images_eval, y_: label_vectors, keep_prob: 0.5})

                if step_num % 20 == 0:
                    # Evaluate train accuracy every 10th step
                    summary, train_accuracy = sess.run([merge_summary, accuracy], feed_dict={x: images_eval, y_: label_vectors, keep_prob: 1.0})
                    train_writer.add_summary(summary, step_num)
                    print("Step: %d Training accuracy: %g" % (step_num, train_accuracy))

                if step_num % 500 == 0:
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


train_CNN()
