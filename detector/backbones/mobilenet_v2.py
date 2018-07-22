import tensorflow as tf
import tensorflow.contrib.slim as slim
from .depthwise_conv import depthwise_conv


BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3


def mobilenet_v2(images, depth_multiplier=1.0, min_depth=8):
    """
    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 255].
        depth_multiplier: a float number, multiplier for the number of filters in a layer.
        min_depth: an integer, the minimal number of filters in a layer.
    Returns:
        a dict with float tensors.
    """

    def depth(x):
        """Reduce the number of filters in a layer."""
        return make_divisible(x * depth_multiplier, divisor=8, min_value=min_depth)

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            training=False, trainable=False,
            fused=True, name='BatchNorm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = ((2.0 / 255.0) * images) - 1.0

    with tf.variable_scope('MobilenetV2'):
        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu6,
            'normalizer_fn': batch_norm,
            'data_format': 'NHWC'
        }
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):

            with slim.arg_scope([slim.conv2d, depthwise_conv], trainable=False):
                x = slim.conv2d(x, depth(32), (3, 3), stride=2, scope='Conv')
                x = inverted_residual_block(
                    x, stride=1, expansion_factor=1,
                    output_channels=depth(16),
                    scope='expanded_conv'
                )

            # (t, c, n, s) - like in the original paper
            block_configs = [
                (6, 24, 2, 2),
                (6, 32, 3, 2),
                (6, 64, 4, 2),
                (6, 96, 3, 1),
                (6, 160, 3, 2),
                (6, 320, 1, 1),
            ]

            i = 1
            features = {}
            for t, c, n, s in block_configs:

                block_name = 'expanded_conv_%d' % i
                x = inverted_residual_block(
                    x, stride=s, expansion_factor=t,
                    output_channels=depth(c), scope=block_name
                )
                features[block_name] = x
                i += 1

                for _ in range(1, n):
                    block_name = 'expanded_conv_%d' % i
                    x = inverted_residual_block(
                        x, stride=1, expansion_factor=t,
                        scope=block_name
                    )
                    features[block_name] = x
                    i += 1

            layer_name = 'Conv_1'
            final_channels = int(1280 * depth_multiplier) if depth_multiplier > 1.0 else 1280
            x = slim.conv2d(x, final_channels, (1, 1), stride=1, scope=layer_name)
            features[layer_name] = x
    features = {}
    return x, features


def inverted_residual_block(x, stride=1, expansion_factor=6, output_channels=None, scope='inverted_residual_block'):

    assert (stride == 1) or (stride == 2 and output_channels is not None)
    in_channels = x.shape.as_list()[1]
    output_channels = output_channels if output_channels is not None else in_channels
    residual = x

    with tf.variable_scope(scope):
        if expansion_factor != 1:
            x = slim.conv2d(
                x, expansion_factor * in_channels, (1, 1),
                stride=1, scope='expand'
            )
        x = depthwise_conv(
            x, kernel=3, stride=stride,
            scope='depthwise'
        )
        x = slim.conv2d(
            x, output_channels, (1, 1),
            stride=1, activation_fn=lambda x: x,
            scope='project'
        )
        return x + residual if in_channels == output_channels and stride == 1 else x


def make_divisible(v, divisor, min_value=None):
    """
    Arguments:
        v: a float.
        divisor, min_value: integers.
    Returns:
        an integer.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, (int(v + divisor / 2) // divisor) * divisor)

    # make sure that round down does not go down by more than 10%
    if new_v < 0.9 * v:
        new_v += divisor

    # now value is divisible by divisor
    # (but not necessarily if min_value is not None)
    return new_v
