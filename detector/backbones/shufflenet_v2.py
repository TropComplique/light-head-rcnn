import tensorflow as tf
import tensorflow.contrib.slim as slim
from .depthwise_conv import depthwise_conv


BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3


def shufflenet(images, depth_multiplier='1.0'):
    """
    This is an implementation of ShuffleNet v2:
    https://arxiv.org/abs/1807.11164

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        depth_multiplier: a string, possible values are '0.5', '1.0', '1.5', and '2.0'.
    Returns:
        a dict with float tensors.
    """
    possibilities = {'0.5': 48, '1.0': 116, '1.5': 176, '2.0': 224}
    initial_depth = possibilities[depth_multiplier]

    # batch norm is frozen and in the inference mode
    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            trainable=False, training=False,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            fused=True, name='batch_norm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = (2.0 * images) - 1.0

    with tf.variable_scope('ShuffleNetV2'):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer()
        }
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):

            # initial layers are not trainable
            with slim.arg_scope([slim.conv2d, depthwise_conv], trainable=False):
                x = slim.conv2d(x, 24, (3, 3), stride=2, scope='Conv1')
                x = slim.max_pool2d(x, (3, 3), stride=2, padding='SAME', scope='MaxPool')  # stride 4

            x = block(x, num_units=4, out_channels=initial_depth, scope='Stage2')  # stride 8
            rpn_features = block(x, num_units=8, scope='Stage3')  # stride 16
            x = block(rpn_features, num_units=4, use_atrous=True, scope='Stage4')

            # in the last stage i don't downsample so i use
            # dilated convolutions to preserve receptive field size

            final_channels = 1024 if depth_multiplier != '2.0' else 2048
            second_stage_features = slim.conv2d(x, final_channels, (1, 1), stride=1, scope='Conv5')  # stride 16

            # if you set `use_atrous=False` in the last
            # stage then the last stride will be 32

    return {'rpn_features': rpn_features, 'second_stage_features': second_stage_features}


def block(x, num_units, out_channels=None, use_atrous=False, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling(
                x, out_channels,
                downsample=not use_atrous
            )

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)
                x = basic_unit(x, rate=1 if not use_atrous else 2)
        x = tf.concat([x, y], axis=3)

    return x


def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):
        shape = tf.shape(x)
        batch_size = x.shape[0].value
        if batch_size is None:
            batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = x.shape[3].value

        z = tf.stack([x, y], axis=3)  # shape [batch_size, height, width, 2, depth]
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y


def basic_unit(x, rate=1):
    in_channels = x.shape[3].value
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    x = depthwise_conv(x, kernel=3, stride=1, rate=rate, activation_fn=None, scope='depthwise')
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_after')
    return x


def basic_unit_with_downsampling(x, out_channels=None, downsample=True):
    in_channels = x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels
    stride = 2 if downsample else 1  # paradoxically, it sometimes doesn't downsample

    y = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    y = depthwise_conv(y, kernel=3, stride=stride, activation_fn=None, scope='depthwise')
    y = slim.conv2d(y, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')

    with tf.variable_scope('second_branch'):
        x = depthwise_conv(x, kernel=3, stride=stride, activation_fn=None, scope='depthwise')
        x = slim.conv2d(x, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')
        return x, y
