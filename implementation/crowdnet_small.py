import tensorflow as tf
import numpy as np
from layers_utils import _conv2d_relu, _avgpool, _maxpool, _activation_summary, _fc_layer
import re
from math import sqrt
import logging

logging.basicConfig()
LOG = logging.getLogger('crowdenet-model')
LOG.setLevel(logging.INFO)


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def put_kernels_on_grid (kernel, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: LOG.info('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return tf.image.convert_image_dtype(x, dtype = tf.uint8)

class CrowdNet():

    def __init__(self):
        self.graph = {}

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.
        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable
        Returns:
          Variable Tensor
        """
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.
        Returns:
          Variable Tensor
        """
        dtype = tf.float32
        var = self._variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var


    def _activation_summary(x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.
        Args:
          x: Tensor
        Returns:
          nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % "tower", '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity',
                          tf.nn.zero_fraction(x))

    def optimizer(self, learning_rate=1e-4):
        with tf.variable_scope('optimizer'):
            """Return optimisation function for neural network
            Don't forget to specify global_step = tf.Variable(0, trainable=False)
            _optimizer(1e-4).minimize(cost_func, global_step=global_step)
            """
            return tf.train.AdamOptimizer(learning_rate)


    def loss(self, logits, labels):
       with tf.variable_scope('loss'):
        """logits = raw final model predictions without softmax or 
        any other kind of probablility calculation, f.e. sigmoid a.ka logistic regression
        """
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss", loss)
        return loss


    def inference(self, input_image_tensor):
        print(input_image_tensor)
        """ define TensorFlow Graph - f.e. VGG architecture
          Use a dictionary to hold the model instead of using a Python class
        """
        """
        VGG-16
        Using average pool instead maxpool to reduce number of parameters
        https://arxiv.org/pdf/1312.4400.pdf
        input_image_tensor - BATCH_SIZE X 512 X 512 X 3 - RGB image
        [batch, height, width, channels].
        """
        tf.summary.image("image_test", input_image_tensor, max_outputs=5)
        # Size:224
        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights', shape=[5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2))
            conv = tf.nn.conv2d(input_image_tensor, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[64],  dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)
            LOG.info("conv1 {} weights/kernel {}, number of params {}".format(conv1.get_shape().as_list(),
                                                                    kernel.get_shape().as_list(),
                                                                    np.prod(kernel.get_shape().as_list()[1:])))
            conv1 = tf.nn.dropout(conv1, keep_prob=0.5)
            scope.reuse_variables()
            weights = tf.get_variable('weights')
            grid = put_kernels_on_grid(kernel=weights)
            tf.summary.image('conv1/kernels', grid, max_outputs=1)
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('weights', shape=[5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2))
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[64],  dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv2)
            LOG.info("conv2 {} weights/kernel {}, number of params {}".format(conv2.get_shape().as_list(),
                                                                    kernel.get_shape().as_list(),
                                                                    np.prod(kernel.get_shape().as_list()[1:])))
            conv2 = tf.nn.dropout(conv2, keep_prob=0.5)
        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],  strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        with tf.variable_scope('fc5') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool2, [128,-1])
            dim = reshape.get_shape()[1].value
            print(dim)
            weights = tf.get_variable('weights', shape=[dim, 192],
                                      initializer=tf.truncated_normal_initializer(stddev=0.04))
            biases  = tf.get_variable('biases', shape=[192], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            fc5 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            _activation_summary(fc5)
            LOG.info("fc5 {}, weights {}, number of params {}".format(fc5.get_shape().as_list(),
                                                            weights.get_shape().as_list(),
                                                            np.prod(weights.get_shape().as_list()[1:])))

            with tf.variable_scope('softmax_linear') as scope:
                weights = tf.get_variable('weights', shape=[192, 2], initializer=tf.truncated_normal_initializer(stddev=1 / 192.0))
                biases = tf.get_variable('biases', shape=[2], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                softmax_linear = tf.add(tf.matmul(fc5, weights), biases, name=scope.name)
                _activation_summary(softmax_linear)
                LOG.info("softmax_linear {}, weights {}, number of params {}".format(softmax_linear.get_shape().as_list(),
                                                                           weights.get_shape().as_list(),
                                                                           np.prod(weights.get_shape().as_list()[1:])))
        return softmax_linear


