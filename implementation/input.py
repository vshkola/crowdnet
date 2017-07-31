from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import numpy as np
import tensorflow as tf

# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/height': tf.FixedLenFeature([],      tf.int64),
            'image/width': tf.FixedLenFeature([],       tf.int64),
            'image/encoded': tf.FixedLenFeature([],     tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (length IMAGE_PIXELS) to
    # a uint8 tensor with shape [IMAGE_PIXELS]
    # image  =  tf.decode_raw(features['image/encoded'], tf.uint8)
    # height =  tf.cast(features['image/height'],        tf.int32)
    # width  =  tf.cast(features['image/width'],         tf.int32)
    # image_shape = tf.stack([height, width, 3])
    # image = tf.reshape(image, image_shape)

    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    # cental mean
    image = tf.cast(image_raw, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['image/class/label'], tf.int32)

    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                           target_height=IMAGE_HEIGHT,
                                                           target_width=IMAGE_WIDTH)
    return resized_image, label


def inputs(train, batch_size, num_epochs):
    """
    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, 1] -> Declined, Accepted.
    tf.train.QueueRunner is added to the graph
  """
    if not num_epochs: num_epochs = None

    train_files = tf.train.match_filenames_once(os.path.join(FLAGS.train_dir,"train-*"))
    validation_files = tf.train.match_filenames_once(os.path.join(FLAGS.train_dir,"validation-*"))

    filenames = train_files if train else validation_files

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

    resized_image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    image_batch, label_batch = tf.train.shuffle_batch(
        [resized_image, label],
        batch_size=batch_size,
        num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=10)

    return image_batch, label_batch


def main(_):
    image_batch, label_batch = inputs(True, 50, num_epochs=FLAGS.num_epochs)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # with tf.Session()  as sess:
    #     sess.run(init_op)
    #
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #
    #     images, labels = sess.run([image_batch, label_batch])
    #
    #     for label in labels:
    #         print(label)
    #
    #     coord.request_stop()
    #     coord.join(threads)

        # Create a session for running operations in the Graph.
    sess = tf.Session()

    # Initialize the variables (the trained variables and the
    # epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        step = 0
        while not coord.should_stop():
            start_time = time.time()
            # Run one step of the model.
            images, labels = sess.run([image_batch, label_batch])
            duration = time.time() - start_time
            # Print an overview fairly often.
            if step % 100 == 0:
                print('Step {}: labels sum = {:0.2f} ({:0.3f} sec)'.format(step, np.sum(labels), duration))
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training for {} epochs, {} steps.'.format(FLAGS.num_epochs, step))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='crowdnet',
        help='Directory with the training data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
