"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm, flatten

slim = tf.contrib.slim

class_num = 5



def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network



def batch_normalization(x, training, scope):
    with slim.arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')

def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net






def densenet(images, num_classes=5, is_training = tf.placeholder(tf.bool),
             dropout_keep_prob=0.8,prediction_fn=slim.softmax,growth = 5,
             nb_blocks=2,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    depth = lambda d: max(int(d * 1), 16)

    def conv_layer(input, filters, kernel_size, stride=1, layer_name="conv"):
        with tf.name_scope(layer_name):
            net = slim.conv2d(input, filters, kernel_size, scope=layer_name)
            return net

    def bottleneck_layer( x, filters, scope):
        with tf.name_scope(scope):
        # [BN --> ReLU --> conv11 --> BN --> ReLU -->conv33]
            x = slim.batch_norm(x)
            x = tf.nn.relu(x)
            x = conv_layer(x, filters, kernel_size=(1, 1), layer_name=scope + '_conv1')
            x = slim.batch_norm(x)
            x = tf.nn.relu(x)
            x = conv_layer(x, filters, kernel_size=(3, 3), layer_name=scope + '_conv2')
            return x


    def dense_block( input_x,  filters,nb_layers, layer_name):
        with tf.name_scope(layer_name) as ssp:
            layers_concat = []
            layers_concat.append(input_x)
            x = bottleneck_layer(input_x, filters, layer_name + '_bottleN_' + str(0))
            layers_concat.append(x)
            for i in range(nb_layers):
                x = tf.concat(layers_concat, axis=3)
                x = bottleneck_layer(x, filters, layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)
        return x

    def transition_layer( x,filters, scope):
        # [BN --> conv11 --> avg_pool2]
        with tf.name_scope(scope):
            x = slim.batch_norm(x)
            x = conv_layer(x, filters, kernel_size=(1, 1), layer_name=scope + '_conv1')
            x = slim.avg_pool2d(x, 2)
        return x

    end_points = {}
    global_pool = True

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            print("----------type(images)------", type(images), "shape:", images.get_shape())
            x = conv_layer(images, growth, kernel_size=(7, 7), layer_name='conv0')

            end_points['conv_layer'] = x
            x = slim.max_pool2d(x, (3, 3))
            end_points['max_pool2d'] = x
            for i in range(nb_blocks):
                #x = dense_block(x, growth, 4, 'dense_' + str(i))
                x = dense_block(x, growth, 4,'dense_' + str(i))
                end_points['dense_' + str(i)] = x
                x = transition_layer(x, growth, 'trans_' + str(i))
                end_points['trans_' + str(i)] = x
                # Auxiliary Head logits
            x = tf.reduce_mean(x, [1, 2], keep_dims=True, name='GlobalPool')
            logits = slim.conv2d(x, num_classes, [1, 1], activation_fn=None,
                                 normalizer_fn=None, scope='Conv2d_1c_1x1')
            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
    return logits, end_points


def bn_drp_scope(is_training, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc

def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.

  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                         tf.minimum(shape[2], kernel_size[1])])

  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out

def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 112

