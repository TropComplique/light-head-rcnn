import tensorflow as tf
import tensorflow.contrib.slim as slim
from .ps_roi_align import position_sensitive_roi_align_pooling


def head(x, rois, roi_image_indices, image_size, params):
    """
    Arguments:
        x: a float tensor with shape [batch_size, height, width, depth].
        rois: a list of float tensors,
            where `i`-th tensor has shape [num_proposals_i, 4],
            and `sum_i num_proposals_i = total_num_proposals`.
        roi_image_indices: an int tensor with shape [total_num_proposals].
        image_size: a tuple of scalar int tensors (image_width, image_height).
        params: a dict.
    Returns:
        encoded_boxes: a float tensor with shape [total_num_proposals, num_classes, 4].
        logits: a float tensor with shape [total_num_proposals, num_classes + 1].
    """

    p, k = params['p'], params['k']
    channels_middle = params['channels_middle']
    num_classes = params['num_classes']
    channels_out = p * p * 10
    # 10 is a value from the original light-head-rcnn paper

    with tf.variable_scope('thin_feature_maps'):
        config = {
            'padding': 'SAME', 'activation_fn': None,
            'weights_initializer': tf.random_normal_initializer(mean=0.0, stddev=0.01)
        }
        with slim.arg_scope([slim.conv2d], **config):
            left = slim.conv2d(x, channels_middle, (k, 1), scope='conv_%dx1_left' % k)
            left = slim.conv2d(left, channels_out, (1, k), scope='conv_1x%d_left' % k)
            right = slim.conv2d(x, channels_middle, (1, k), scope='conv_1x%d_right' % k)
            right = slim.conv2d(right, channels_out, (k, 1), scope='conv_%dx1_right' % k)
            x = tf.nn.relu(left + right)  # it has the same spatial size as initial `x`

    with tf.name_scope('position_sensitive_roi_align'):
        rois = tf.concat(rois, axis=0)  # shape [total_num_proposals, 4]

        # convert to normalized coordinates
        image_width, image_height = image_size
        scaler = tf.to_float(tf.stack([
            image_height, image_width,
            image_height, image_width
        ]))
        rois = rois/scaler

        x = position_sensitive_roi_align_pooling(
            x, rois, roi_image_indices,
            crop_size=(2*p, 2*p),  # (14, 14)
            num_spatial_bins=(p, p)  # (7, 7)
        )  # shape [total_num_proposals, p * p, 10]

    with tf.variable_scope('fc_layers'):

        # flatten
        total_num_proposals = tf.shape(x)[0]
        x = tf.reshape(x, [total_num_proposals, p * p * 10])

        x = slim.fully_connected(
            x, params['fc_layer_size'],
            activation_fn=tf.nn.relu, scope='large_fc'
        )
        encoded_boxes = slim.fully_connected(
            x, 4 * num_classes,
            weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
            activation_fn=None,  scope='boxes'
        )
        logits = slim.fully_connected(
            x, num_classes + 1,
            weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            activation_fn=None, scope='classes'
        )

        encoded_boxes = tf.reshape(encoded_boxes, [total_num_proposals, num_classes, 4])
        return encoded_boxes, logits
