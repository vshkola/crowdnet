from __future__ import print_function

from crowdnet_small import CrowdNet
from crowdnet_input import inputs, input_dataset
from crowdnet_read_dataset import read_dataset
from layers_utils import _accuracy, _correct_pred

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import argparse
import sys
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops

logging.basicConfig()
LOG = logging.getLogger('crowdenet-train')
LOG.setLevel(logging.INFO)

# Define paramaters for the model
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_OF_EPOCHS = 100

# Network Parameters
N_CLASS = 2

def train(images, labels, opt, loss_op, merged_summaries, global_step, top_k_op, logits, agent_ids, accuracy, errors, abs_errors):
    LOG.info("Training started.")
    # Create a session for running operations in the Graph.
    with tf.Session() as sess:
        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        LOG.info("global and local variables initialized")
        # Initialize the variables (the trained variables and the epoch counter
        sess.run(init_op)
        # Define saving checkpoint for the model
        # Define graph writer for the model, for further visualisation with TensorBoard
        #LOG.info("Setup saver and writer")
        # saver = tf.train.Saver()
        writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
        LOG.info("To see summaries, please open http://localhost:6006")
        # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        epoch = 0
        try:
           LOG.info("start coordinator and threading")
           while not coord.should_stop() or epoch >= NUM_OF_EPOCHS:
                start_time = time.time()
                for step in range(50):
                    start_time = time.time()
                    # Run one step of the model.
                    #agent = sess.run([agent_ids])
                   # LOG.info("Number of agents in a batch is {}".format(len(agent)))
                    _, loss, summary, _ = sess.run([opt, loss_op, merged_summaries])
                    writer.add_summary(summary, global_step=epoch)
                    #train_accuracy = "by top_k_op {:.3%} \nby metrics.accuracy {}".format(np.sum(predictions) / float(np.size(predictions)), accuracy)
                    LOG.info("Step {0} is finished in {1:.2f}s. loss - {2}".format(step, time.time() - start_time, loss))
                accuracy_scalar, errors_scalar, abs_error_scalar = sess.run([accuracy, errors, abs_errors])
                duration = time.time() - start_time
                LOG.info("Epoch {0} is finished in {1:.2f}s.".format(epoch, duration))
                if epoch % 5 == 0:
                    #saver.save(sess, save_path=os.path.join('checkpoints', 'checkpoint_{}'.format(str(epoch))))
                     LOG.info('\nepoch {}\nloss - {}\ntrain accuracy - {}\nerrors - {}\n - abs errors {}'.format(epoch, loss, accuracy_scalar[1], errors_scalar[1], abs_error_scalar))
                     #print(sess.run(logits))
                     #for gv in grads:
                     #   print(str(sess.run(gv[0])) + " - " + gv[1].name)
                epoch += 1
        except tf.errors.OutOfRangeError:
            LOG.info('Done training for {} epochs'.format(NUM_OF_EPOCHS))
        finally:
            coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        writer.close()

def _create_summary(model, loss_fn):
    """ Create summary ops necessary
    """
    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', loss_fn)
#        tf.summary.scalar('accuracy',   model['accuracy'])
        tf.summary.histogram('histogram loss', loss_fn)
        return tf.summary.merge_all()

def main(_):
    read_dataset()

    crowdnet = CrowdNet()
    logits = crowdnet.inference(images)

    global_step = tf.contrib.framework.get_or_create_global_step()
    loss = crowdnet.loss(logits=logits, labels=labels)

    #learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 100000, 0.96, staircase=True)
    opt = tf.train.AdamOptimizer(0.001)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step, name='train_op')

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    #LOG.info(tf.argmax(logits,1).shape)
    #LOG.info(tf.argmax(labels,1).shape)

    top_k_op   = tf.nn.in_top_k(logits,      labels,      1)

    accuracy = tf.metrics.accuracy(predictions=tf.argmax(logits,1), labels=labels)
    errors = tf.metrics.mean_absolute_error(predictions=tf.cast(tf.argmax(logits,1), tf.int64), labels=tf.cast(labels, tf.int64))
    abs_errors= math_ops.reduce_sum(math_ops.abs(tf.cast(tf.argmax(logits,1), tf.int64) - tf.cast(labels, tf.int64)))

    merged_summaries_op = tf.summary.merge_all()
    train(images, labels, apply_gradient_op, loss, merged_summaries_op, global_step, top_k_op, logits, agent_ids, accuracy, errors, abs_errors, iterator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='crowdnet_test',
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--summary_dir',
        type=str,
        default="graphs/crowdnet",
        help='Directory with summaries'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

