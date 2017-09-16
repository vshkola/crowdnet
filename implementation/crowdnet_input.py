from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import numpy as np
import tensorflow as tf
import re, logging

logging.basicConfig()
LOG = logging.getLogger('crowdenet-input')
LOG.setLevel(logging.INFO)

# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# Global constants describing the Crowdnet data set.
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

DATASET_BUFFER_SIZE = 10000

def read_and_decode(filename_queue, with_transformation):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'agent_id' :     tf.FixedLenFeature([],     tf.string),
            'image/height':  tf.FixedLenFeature([],     tf.int64),
            'image/width':   tf.FixedLenFeature([],     tf.int64),
            'image/encoded': tf.FixedLenFeature([],     tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/label_complex': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (length IMAGE_PIXELS) to
    # a uint8 tensor with shape [IMAGE_PIXELS]
    # image  =  tf.decode_raw(features['image/encoded'], tf.uint8)
    # height =  tf.cast(features['image/height'],        tf.int32)
    # width  =  tf.cast(features['image/width'],         tf.int32)
    # image_shape = tf.stack([height, width, 3])
    # image = tf.reshape(image, image_shape)
    agent_id = features["agent_id"]

    image_encoded = features["image/encoded"]

    raw_image = tf.image.decode_jpeg(image_encoded, channels=3)
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['image/class/label'], tf.int32)

    if with_transformation:
        LOG.info("Images will be resized to {} x {}".format(IMAGE_HEIGHT, IMAGE_WIDTH))
        resized_image = tf.image.resize_image_with_crop_or_pad(image=raw_image,
                                                               target_height=IMAGE_HEIGHT,
                                                               target_width=IMAGE_WIDTH)
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        # central mean
        resized_image = tf.cast(resized_image, tf.float32) * (1. / 255) - 0.5

        transformed_image = tf.image.per_image_standardization(resized_image)
        return (transformed_image, label, agent_id)
    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(image=raw_image,
                                                               target_height=512,
                                                               target_width=512)
    return (resized_image, label, agent_id)


def inputs(dataset_type, batch_size, num_epochs, train_dir, with_transformation=True):
    """
    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, 1] -> Declined, Accepted.
    tf.train.QueueRunner is added to the graph
  """
    IS_TRAIN = 0
    IS_TEST  = 1
    IS_EVAL  = 2

    LOG.info("Produce input for {}".format("training" if dataset_type == IS_TRAIN else "test/validation"))

    if not num_epochs: num_epochs = None

    train_files = tf.train.match_filenames_once(os.path.join(train_dir,"train-*"))
    test_files  = tf.train.match_filenames_once(os.path.join(train_dir, "test-*"))
    validation_files = tf.train.match_filenames_once(os.path.join(train_dir,"validation-*"))

    if dataset_type == IS_TRAIN: filenames = train_files
    if dataset_type == IS_TEST:  filenames = test_files
    if dataset_type == IS_EVAL:  filenames = validation_files

    with tf.name_scope('input'):
        LOG.info("Num epochs for input: {}".format(num_epochs))
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

    resized_image, label, agent_id = read_and_decode(filename_queue, with_transformation)

    check_num_of_images_in_tfrecords(train_dir)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    LOG.info("Produce shuffled batch with {} batch size".format(batch_size))
    image_batch, label_batch, agent_batch = tf.train.shuffle_batch(
        [resized_image, label, agent_id],
        batch_size=batch_size,
        num_threads=4,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)
    tf.summary.image('images', image_batch)
    return (image_batch, label_batch, agent_batch)

def input_dataset(dataset_type, num_of_epochs, batch_size, train_dir):

    '''
    Help-function to parse each jpeg image
    :param dataset_type:
    :param batch_size:
    :param train_dir:
    :return:
    '''
    def _parse_function(example_proto):
        features = {"image/encoded":     tf.FixedLenFeature([], tf.string),
                    "image/class/label": tf.FixedLenFeature([], tf.int64),
                    "agent_id" :         tf.FixedLenFeature([], tf.string),
                    }
        parsed_features = tf.parse_single_example(example_proto, features)

        image_encoded = parsed_features["image/encoded"]
        raw_image = tf.image.decode_jpeg(image_encoded, channels=3)
        raw_image = tf.cast(raw_image, tf.float32)
        label = parsed_features["image/class/label"]
        agent = parsed_features["agent_id"]

        resized_image = tf.image.resize_image_with_crop_or_pad(image=raw_image,
                                                               target_height=IMAGE_HEIGHT,
                                                               target_width=IMAGE_WIDTH)
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        # central mean
        resized_image = tf.cast(resized_image, tf.float32) * (1. / 255) - 0.5

        transformed_image = tf.image.per_image_standardization(resized_image)

        return (transformed_image, label, agent)
    """
    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, 1] -> Declined, Accepted.
    tf.train.QueueRunner is added to the graph
  """
    IS_TRAIN = 0
    IS_TEST  = 1
    IS_EVAL  = 2

    LOG.info("Produce input for {}".format("training" if dataset_type == IS_TRAIN else "test/validation"))

    train_files = tf.train.match_filenames_once(os.path.join(train_dir,"train-*"))
    test_files  = tf.train.match_filenames_once(os.path.join(train_dir, "test-*"))
    validation_files = tf.train.match_filenames_once(os.path.join(train_dir,"validation-*"))

    check_num_of_images_in_tfrecords(train_dir)

    if dataset_type == IS_TRAIN: filenames = train_files
    if dataset_type == IS_TEST:  filenames = test_files
    if dataset_type == IS_EVAL:  filenames = validation_files

    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=DATASET_BUFFER_SIZE)
    dataset = dataset.repeat(num_of_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    next_batch = iterator.get_next()

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    LOG.info("Produce shuffled batch with {} batch size".format(batch_size))

    return iterator_init_op, next_batch

def check_num_of_images_in_tfrecords(train_dir):
    for t in ["train", "test", "validation"]:
        tfrecords_files = [os.path.abspath(os.path.join(train_dir, f)) for f in os.listdir(train_dir) if re.match(r'{}-*'.format(t), f)]
        c = 0
        for fn in tfrecords_files:
            for _ in tf.python_io.tf_record_iterator(fn):
                c += 1
        LOG.info("{} samples in total: {}".format(t, c))