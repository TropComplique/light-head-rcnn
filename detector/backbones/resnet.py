import tensorflow as tf
import tf.contrib.slim as slim
from detector.constants import BATCH_NORM_MOMENTUM, BATCH_NORM_EPSILON


"""
Notes:

1. This implementation is taken from here:
   https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py

2. It differs from the original resnet architecture (like in the original paper).
   In this implementation we subsample the output activations in the last residual unit of
   each block, instead of subsampling the input activations in the first residual
   unit of each block. The two implementations give identical results but our
   implementation is more memory efficient

3. In the fourth block

4. The spatial size of the third block output is given by:
   height = ceil(image_height/16)
   width = ceil(image_width/16)

1.  Training for image classification on Imagenet is usually done with [224, 224]
    inputs, resulting in [7, 7] feature maps at the output of the last ResNet
    block for the ResNets defined in [1] that have nominal stride equal to 32.
    However, for dense prediction tasks we advise that one uses inputs with
    spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
    this case the feature maps at the ResNet output will have spatial shape
    [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
    and corners exactly aligned with the input image corners, which greatly
    facilitates alignment of the features to the image. Using as input [225, 225]
    images results in [8, 8] feature maps at the output of the last ResNet block.

5. First layers are not trainable
6. Batch norm layers are not updated
Dilated
"""


def resnet(images, is_training):
    """This is classical ResNet-50 architecture.

    It implemented in a way that works with
    the official tensorflow pretrained checkpoints.

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            it represents RGB images with pixel values in range [0, 255].
        is_training: a boolean.
    Returns:
        a dict of with two float tensors.
    """

    with tf.name_scope('standardize'):
        channel_means = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
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

            # weights of these layers are fixed during training
            with slim.arg_scope([slim.conv2d], trainable=False):
                x = conv2d_same(x, 64, 7, stride=2, scope='conv1')
                x = slim.max_pool2d(x, [3, 3], stride=2, padding='SAME', scope='pool1')
                x = block(x, base_depth=64, num_units=3, stride=2, scope='block1')

            x = block(
                x, base_depth=128, num_units=4,
                stride=2, scope='block2'
            )
            rpn_features = block(
                x, base_depth=256, num_units=6,
                stride=2, scope='block3'
            )
            second_stage_features = block(
                rpn_features, base_depth=512, num_units=3,
                stride=1, rate=2, scope='block4'
            )

        return {'block3': rpn_features, 'block4': second_stage_features}


def block(x, base_depth, num_units, stride, rate=1, scope='block'):
    for i in range(1, num_units):
        x = bottleneck(
            x, depth=base_depth * 4, depth_bottleneck=base_depth,
            stride=1, rate=rate, scope='unit_%d' % i
        )
    x = bottleneck(
        x, depth=base_depth * 4, depth_bottleneck=base_depth,
        stride=stride, rate=rate, scope='unit_%d' % num_units
    )
    return x


def bottleneck(x, depth, depth_bottleneck, stride, rate=1, scope='bottleneck'):
    with tf.variable_scope(scope):

        residual = slim.conv2d(x, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')

        depth_in = x.shape.as_list()[3]
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
