import tensorflow as tf
import tensorflow.contrib.slim as slim


BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-5


"""
Notes:
1. This implementation is taken from here:
   https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py

2. It differs from the original resnet architecture (like in the original paper).
   In this implementation I subsample the output activations in the last residual unit of
   each block, instead of subsampling the input activations in the first residual
   unit of each block.

3. In the fourth block I use dilation rate = 2 in all units.

4. Batch norm layers are not updated during training.
   They are in the inference mode.

5. I don't use stride in the last unit of the third block.
   Instead I use dilation rate = 2.

6. The first layers of the network are frozen during training.

7. The spatial size of the third block output is given by:
   height, width = ceil(image_height/16), ceil(image_width/16).
"""


def resnet(images):
    """
    This is classical ResNet-50 architecture
    (with some changes noted above).

    It implemented in a way that works with
    the official tensorflow pretrained checkpoints.

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            it represents RGB images with pixel values in range [0, 255].
    Returns:
        a dict of with two float tensors.
    """

    with tf.name_scope('standardize'):
        channel_means = tf.constant([123.15163, 115.902885, 103.06262], dtype=tf.float32)
        x = images - channel_means

    def batch_norm(x):
        x = tf.layers.batch_normalization(
           x, axis=3, center=True, scale=True,
           momentum=BATCH_NORM_MOMENTUM,
           epsilon=BATCH_NORM_EPSILON,
           training=False, trainable=False,
           fused=True, name='BatchNorm'
        )
        return x

    with tf.variable_scope('resnet_v1_50'):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=batch_norm):

            # this will be used by the detector
            features = {}

            # a resnet unit is represented by a tuple:
            # (base_depth, stride, rate)

            # weights of these layers are fixed during training
            with slim.arg_scope([slim.conv2d], trainable=False):
                x = conv2d_same(x, 64, 7, stride=2, scope='conv1')
                x = slim.max_pool2d(x, [3, 3], stride=2, padding='SAME', scope='pool1')
                x = stack_units(x, [(64, 1, 1)] * 2 + [(64, 2, 1)], scope='block1')
                x = stack_units(x, [(128, 1, 1)] * 3 + [(128, 2, 1)], scope='block2')

            x = stack_units(x, [(256, 1, 1)] * 5 + [(256, 1, 2)], scope='block3')
            features['rpn_features'] = x

            x = stack_units(x, [(512, 1, 2)] * 3, scope='block4')
            features['second_stage_features'] = x

        return features


def stack_units(x, config, scope='block'):
    with tf.variable_scope(scope):
        num_units = len(config)
        for i in range(num_units):
            base_depth, stride, rate = config[i]
            x = bottleneck(
                x, depth=base_depth * 4,
                depth_bottleneck=base_depth,
                stride=stride, rate=rate,
                scope='unit_%d/bottleneck_v1' % (i + 1)
            )
        return x


def bottleneck(x, depth, depth_bottleneck, stride, rate=1, scope='bottleneck'):
    with tf.variable_scope(scope):

        residual = slim.conv2d(x, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')

        depth_in = x.shape[3].value
        if depth == depth_in:
            shortcut = subsample(x, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(x, depth, [1, 1], stride=stride, activation_fn=None, scope='shortcut')

        return tf.nn.relu(shortcut + residual)


def subsample(x, factor, scope):
    return x if factor == 1 else slim.max_pool2d(x, [1, 1], stride=factor, scope=scope)


def conv2d_same(x, num_outputs, kernel_size, stride, rate=1, scope='conv2d_same'):
    if stride == 1:
        return slim.conv2d(
            x, num_outputs, kernel_size, stride=1, rate=rate,
            padding='SAME', scope=scope
        )
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(
            x, num_outputs, kernel_size, stride=stride, rate=rate,
            padding='VALID', scope=scope
        )
