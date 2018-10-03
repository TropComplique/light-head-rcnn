import tensorflow as tf
import tensorflow.contrib.slim as slim
from detector.constants import BATCH_NORM_MOMENTUM, BATCH_NORM_EPSILON, DATA_FORMAT
from .depthwise_conv import depthwise_conv


def mobilenet_v1(images, is_training, depth_multiplier=1.0):
    """
    Arguments:
        images: a float tensor with shape [batch_size, height, width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        depth_multiplier: a float number, multiplier for the number of filters in a layer.
    Returns:
        a dict with float tensors.
    """

    def depth(x):
        """Reduce the number of filters in a layer."""
        return max(int(x * depth_multiplier), 8)

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=1 if DATA_FORMAT == 'channels_first' else 3,
            center=True, scale=True,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            training=is_training, fused=True,
            name='BatchNorm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = (2.0 * images) - 1.0

    with tf.variable_scope('MobilenetV1'):
        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu6, 'normalizer_fn': batch_norm,
            'data_format': 'NCHW' if DATA_FORMAT == 'channels_first' else 'NHWC'
        }
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):
            features = {}

            if DATA_FORMAT == 'channels_first':
                x = tf.transpose(x, [0, 3, 1, 2])

            layer_name = 'Conv2d_0'
            x = slim.conv2d(x, depth(32), (3, 3), stride=2, scope=layer_name)
            features[layer_name] = x

            strides_and_filters = [
                (1, 64),
                (2, 128), (1, 128),
                (2, 256), (1, 256),
                (2, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512),
                (2, 1024), (1, 1024)
            ]
            for i, (stride, num_filters) in enumerate(strides_and_filters, 1):

                layer_name = 'Conv2d_%d_depthwise' % i
                x = depthwise_conv(x, stride=stride, scope=layer_name)
                features[layer_name] = x

                layer_name = 'Conv2d_%d_pointwise' % i
                x = slim.conv2d(x, depth(num_filters), (1, 1), stride=1, scope=layer_name)
                features[layer_name] = x

    return {
        'c3': features['Conv2d_5_pointwise'],
        'c4': features['Conv2d_11_pointwise'],
        'c5': features['Conv2d_13_pointwise']
    }
