import tensorflow as tf
import numpy as np
import re
from layers_utils import _conv2d_relu, _avgpool, _maxpool, _activation_summary, _fc_layer
from math import sqrt



class CrowdnetModel():

    def __init__(self):
        self.graph = {}


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
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=labels)
        return tf.reduce_mean(loss)

    def accuracy(self, pred, ):
        with tf.name_scope('Accuracy'):
         # Accuracy
            acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))


    def define_graph(self, input_image_tensor):
        """ define TensorFlow Graph - f.e. VGG architecture
          Use a dictionary to hold the model instead of using a Python class
        """
        graph = {}

        """
        VGG-16
        Using average pool instead maxpool to reduce number of parameters
        https://arxiv.org/pdf/1312.4400.pdf
        input_image_tensor - BATCH_SIZE X 512 X 512 X 3 - RGB image
        [batch, height, width, channels].
        """
        # Size:224
        graph['conv1_1'] = _conv2d_relu(input_image_tensor, [5, 5, 3, 64], 'conv1_1')
        graph['conv1_2'] = _conv2d_relu(graph['conv1_1'],   [5, 5, 64, 64], 'conv1_2')
        #_activation_summary(graph['conv1_2'])
        graph['maxpool1'] = _maxpool(graph['conv1_2'], ksize=[1, 3, 3, 1], name='maxpool1')
        # Size:112
        graph['conv2_1'] = _conv2d_relu(graph['maxpool1'], [3, 3, 64, 128], 'conv2_1')
        graph['conv2_2'] = _conv2d_relu(graph['conv2_1'],  [3, 3, 128, 128], 'conv2_2')
        #_activation_summary(graph['conv2_2'])
        graph['maxpool2'] = _maxpool(graph['conv2_2'], ksize=[1, 3, 3, 1], name='maxpool2')
        # Size 56
        with tf.variable_scope('fc5') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(graph['maxpool1'], [128,-1])
            dim = reshape.get_shape()[1].value
            weights = tf.get_variable('weights', shape=[dim, 192],
                                      initializer=tf.truncated_normal_initializer(stddev=0.04))
            biases  = tf.get_variable('biases',shape=[192], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            fc5 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            print("fc5", fc5.get_shape().as_list(), "fc5" + "_w", weights.get_shape().as_list(), "params",
                  np.prod(weights.get_shape().as_list()[1:]))
       #     _activation_summary(fc5)
        graph['fc6'] = _fc_layer(fc5, in_size=384, out_size=192, name='fc6')
        graph['fc7'] = _fc_layer(graph['fc6'], in_size=192, out_size=192, name='fc7')
        graph['conv3_1'] = _conv2d_relu(graph['avgpool2'], [3, 3, 128, 256], 'conv3_1')
        graph['conv3_2'] = _conv2d_relu(graph['conv3_1'],  [3, 3, 256, 256], 'conv3_2')
        graph['conv3_3'] = _conv2d_relu(graph['conv3_2'],  [3, 3, 256, 256], 'conv3_3')
        # _activation_summary(graph['conv3_3'])
        graph['avgpool3'] = _maxpool(graph['conv3_3'],  ksize=[1, 2, 2, 1], name='avg_pool_3')
        #
        graph['conv4_1'] = _conv2d_relu(graph['avgpool3'], [3, 3, 256, 512], 'conv4_1')
        graph['conv4_2'] = _conv2d_relu(graph['conv4_1'],  [3, 3, 512, 512], 'conv4_2')
        graph['conv4_3'] = _conv2d_relu(graph['conv4_2'],  [3, 3, 512, 512], 'conv4_3')
        # _activation_summary(graph['conv4_3'])
        graph['avgpool4'] = _maxpool(graph['conv4_3'],  ksize=[1, 2, 2, 1], name='avg_pool_4')
        #
        graph['conv5_1'] = _conv2d_relu(graph['avgpool4'], [3, 3, 512, 512], 'conv5_1')
        graph['conv5_2'] = _conv2d_relu(graph['conv5_1'],  [3, 3, 512, 512], 'conv5_2')
        graph['conv5_3'] = _conv2d_relu(graph['conv5_2'],  [3, 3, 512, 512], 'conv5_3')
        # _activation_summary(graph['conv5_3'])
        # # logits
        graph['avgpool5'] = _avgpool(graph['conv5_3'], ksize=[1, 2, 2, 1], name='avg_pool_5')
        #_activation_summary(graph['fc7'])
        self.graph = graph

        return (graph)

    def put_kernels_on_grid(kernel, pad=1):

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
                    if i == 1: print('Who would enter a prime number of filters')
                    return (i, int(n / i))

        (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
        print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        kernel = (kernel - x_min) / (x_max - x_min)

        # pad X and Y
        x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

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
        return x

