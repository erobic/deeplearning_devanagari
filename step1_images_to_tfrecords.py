import os
import proj_constants
import tensorflow as tf
import numpy as np
import threading

MAX_EXAMPLES_PER_FILE = 5


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def examples_to_tfrecords(image_files, labels, tfrecords_file):
    '''Saves image files and their labels into a single tfrecords file.
    Each record has "x" which is the gray-scale image having intensity values between 0 and 1
    and "y" which is the vector with "CLASSES" no. of entries with all entries being 0 except the actual label
    which is set to 1'''
    file_queue = tf.train.string_input_producer(image_files)
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)
    decoded_img = tf.image.decode_png(value, channels=1)

    init_op = tf.initialize_all_variables()

    writer = tf.python_io.TFRecordWriter(tfrecords_file)

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in xrange(len(image_files)):
            image = decoded_img.eval()
            image = image*(1./255.)
            image_raw = image.tostring()

            #label_vector = to_label_vector(labels[i])
            #label_vector_raw = label_vector.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'x': _bytes_feature(image_raw),
                'y': _int64_feature(int(labels[i]))
            }))
            writer.write(example.SerializeToString())

        coord.request_stop()
        coord.join(threads)


def charfolder_to_tfrecords(label_dir, images, output_dir):
    img_cnt = 0
    image_files = []
    labels = []
    label_name = label_dir.split(os.sep)[2]
    label = proj_constants.get_label_id(label_name)
    master_filename = os.path.join(output_dir, label_name)
    for img in images:
        image_file = os.path.join(label_dir, img)
        image_files.append(image_file)
        labels.append(label)
        if img_cnt != 0 and ((img_cnt + 1) % MAX_EXAMPLES_PER_FILE == 0 or (img_cnt == len(images) - 1)):
            filename = master_filename + "_" + str(img_cnt + 1) + ".tfrecords"
            print("Writing to: %s" %filename)
            thr = threading.Thread(target=examples_to_tfrecords, args=(image_files, labels, filename), kwargs={})
            thr.start()
            image_files = []
            labels = []
        img_cnt += 1


def datafolder_to_tfrecords(data_folder, output_dir):
    '''Retrieves image and label info from given data_folder and saves them to a tfrecords file'''
    walk = os.walk(data_folder)
    i = 0
    for x in walk:
        if i != 0:
            label_dir = x[0]
            images = x[2]
            thr = threading.Thread(target=charfolder_to_tfrecords, args=(label_dir, images, output_dir), kwargs={})
            thr.start()
        i += 1

TFRECORDS_TRAIN_DIR = os.path.join(proj_constants.DATA_DIR, 'tfrecords', 'train')
TFRECORDS_TEST_DIR = os.path.join(proj_constants.DATA_DIR, 'tfrecords', 'test')
datafolder_to_tfrecords(proj_constants.TRAIN_DIR, TFRECORDS_TRAIN_DIR)
datafolder_to_tfrecords(proj_constants.TEST_DIR, TFRECORDS_TEST_DIR)