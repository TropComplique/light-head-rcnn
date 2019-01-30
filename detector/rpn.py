import tensorflow as tf
import tensorflow.contrib.slim as slim
import itertools
from detector.utils import prune_outside_window, clip_by_window, batch_decode


"""
Notes:
1. All boxes are in absolute coordinates.
   `ymin` and `ymax` are float numbers in [0, image_height] range.
   `xmin` and `xmax` are float numbers in [0, image_width] range.
   And for each box I define:
   height = ymax - ymin, width = xmax - xmin, aspect_ratio = width/height.
   Also it must be that:
   ymin < ymax and xmin < xmax.

2. Scale of an anchor box is defined like this:
   scale = sqrt(height * width).

3. Total number of anchor boxes depends on the image size.

4. `scale` and `aspect_ratio` are independent of the image size.

5. The center of the top-left anchor is [0, 0].
   It means that generated anchors are not
   necessary symmetrically distributed on the image.
   Is this a problem?
"""


def rpn(x, is_training, image_size, params):
    """Predicts boxes, generates anchors, and postprocesses proposals.

    Arguments:
        x: a float tensor with shape [batch_size, height, width, depth],
            features from a backbone network.
        is_training: a boolean.
        image_size: a tuple of scalar int tensors (image_width, image_height).
        params: a dict.
    Returns:
        proposals: a dict with the following keys
            'rois': a list of float tensors,
                where `i`-th tensor has shape [num_proposals_i, 4],
                and `sum_i num_proposals_i = total_num_proposals`.
            'roi_image_indices': an int tensor with shape [total_num_proposals].
            'num_proposals_per_image': an int tensor with shape [batch_size].
        encoded_boxes: a float tensor with shape [batch_size, num_anchors, 4].
        objectness_scores: a float tensor with shape [batch_size, num_anchors, 2].
        anchors: a float tensor with shape [num_anchors, 4].
    """
    scales, aspect_ratios = params['scales'], params['aspect_ratios']
    num_anchors_per_cell = len(scales) * len(aspect_ratios)

    with tf.variable_scope('rpn'):
        x = slim.conv2d(
            x, params['rpn_num_channels'], (3, 3), padding='SAME',
            activation_fn=tf.nn.relu, scope='conv'
        )
        raw_encoded_boxes = slim.conv2d(
            x, num_anchors_per_cell * 4, (1, 1),
            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.001),
            activation_fn=None, scope='bounding_boxes'
        )
        raw_objectness_scores = slim.conv2d(
            x, num_anchors_per_cell * 2, (1, 1),
            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            activation_fn=None, scope='objectness_scores'
        )

    height, width = tf.shape(x)[1], tf.shape(x)[2]  # feature map size
    raw_anchors = generate_anchors(
        grid_size=(width, height),
        scales=scales, aspect_ratios=aspect_ratios,
        anchor_stride=params['anchor_stride'],
        anchor_offset=params['anchor_offset']
    )
    # anchors are raw because they are non pruned and non clipped

    all_proposals, rpn_output, anchors = postprocess_and_get_proposals(
        raw_encoded_boxes, raw_objectness_scores, raw_anchors,
        num_anchors_per_cell=num_anchors_per_cell,
        is_training=is_training, image_size=image_size
    )
    proposals = choose_proposals(all_proposals, params)

    return proposals, rpn_output, anchors


def generate_anchors(
        grid_size, scales=[32, 64, 128, 256, 512],
        aspect_ratios=[0.5, 1.0, 2.0],
        anchor_stride=(16, 16), anchor_offset=(0, 0)):
    """
    Arguments:
        grid_size: a tuple of scalar int tensors (grid_width, grid_height).
        scales: a list of integers with length n.
        aspect_ratios: a list of floats with length m.
        anchor_stride: a tuple of integers, difference in centers between
            anchors for adjacent grid positions.
        anchor_offset: a tuple of integers,
            center of the anchor on upper left element of the grid ((0, 0)-th anchor).
    Returns:
        a float tensor with shape [grid_height, grid_width, n * m, 4].
    """
    with tf.name_scope('generate_anchors'):
        pairs = list(itertools.product(scales, aspect_ratios))
        N = len(pairs)  # n * m
        grid_width, grid_height = grid_size

        scales = tf.constant([s for s, _ in pairs], dtype=tf.float32)
        aspect_ratios = tf.constant([a for _, a in pairs], dtype=tf.float32)
        ratios_sqrt = tf.sqrt(aspect_ratios)
        heights = scales / ratios_sqrt
        widths = scales * ratios_sqrt

        stride_y, stride_x = anchor_stride
        y_translation = tf.to_float(tf.range(grid_height)) * stride_y
        x_translation = tf.to_float(tf.range(grid_width)) * stride_x
        x_translation, y_translation = tf.meshgrid(x_translation, y_translation)
        # they have shape [grid_height, grid_width]

        center_y, center_x = anchor_offset
        center_translations = tf.stack([y_translation, x_translation], axis=2)
        centers = tf.constant([center_y, center_x], dtype=tf.float32) + center_translations
        # they have shape [grid_height, grid_width, 2]

        sizes = tf.stack([heights, widths], axis=1)  # shape [N, 2]
        sizes = tf.expand_dims(tf.expand_dims(sizes, 0), 0)
        sizes = tf.tile(sizes, [grid_height, grid_width, 1, 1])
        # shape [grid_height, grid_width, N, 2]

        centers = tf.expand_dims(centers, 2)  # shape [grid_height, grid_width, 1, 2]
        centers = tf.tile(centers, [1, 1, N, 1])  # shape [grid_height, grid_width, N, 2]

        # to [ymin, xmin, ymax, xmax] format
        cy, cx = tf.unstack(centers, axis=3)
        h, w = tf.unstack(sizes, axis=3)
        return tf.stack([cy - 0.5*h, cx - 0.5*w, cy + 0.5*h, cx + 0.5*w], axis=3)


def postprocess_and_get_proposals(
        encoded_boxes, objectness_scores,
        anchors, num_anchors_per_cell,
        is_training, image_size):
    """
    Arguments:
        encoded_boxes: a float tensor with shape [batch_size, height, width, 4 * N].
        objectness_scores: a float tensor with shape [batch_size, height, width, 2 * N].
        anchors: a float tensor with shape [height, width, N, 4].
        num_anchors_per_cell: an integer, it equals to N.
        is_training: a boolean.
        image_size: a tuple of scalar int tensors (image_width, image_height).
    Returns:
        boxes: a float tensor with shape [batch_size, num_anchors, 4].
        probabilities: a float tensor with shape [batch_size, num_anchors].
        encoded_boxes: a float tensor with shape [batch_size, num_anchors, 4].
        objectness_scores: a float tensor with shape [batch_size, num_anchors, 2].
        anchors: a float tensor with shape [num_anchors, 4].
    """
    batch_size = encoded_boxes.shape[0].value
    height, width = tf.shape(anchors)[0], tf.shape(anchors)[1]
    image_width, image_height = image_size
    window = tf.to_float(tf.stack([0, 0, image_height, image_width]))

    with tf.name_scope('reshaping'):

        # convert to a shape that looks like `anchors` shape
        encoded_boxes = tf.reshape(encoded_boxes, [batch_size, height, width, num_anchors_per_cell, 4])
        objectness_scores = tf.reshape(objectness_scores, [batch_size, height, width, num_anchors_per_cell, 2])

        # it is important to reshape all these tensors in the same way
        anchors = tf.reshape(anchors, [-1, 4])
        encoded_boxes = tf.reshape(encoded_boxes, [batch_size, -1, 4])
        objectness_scores = tf.reshape(objectness_scores, [batch_size, -1, 2])

    # see the original faster-rcnn paper about pruning and clipping
    if is_training:
        anchors, valid_indices = prune_outside_window(anchors, window)
        encoded_boxes = tf.gather(encoded_boxes, valid_indices, axis=1)
        objectness_scores = tf.gather(objectness_scores, valid_indices, axis=1)
    else:
        # note: it is assumed that all anchors have
        # nonzero intersection with the window
        anchors = clip_by_window(anchors, window)
        # do i need to clip here or not?

    probabilities = tf.nn.softmax(objectness_scores, axis=2)  # shape [batch_size, num_anchors, 2]
    probabilities = probabilities[:, :, 1]  # shape [batch_size, num_anchors]
    boxes = batch_decode(encoded_boxes, anchors)  # shape [batch_size, num_anchors, 4]
    boxes = clip_by_window(boxes, window)  # shape [batch_size, num_anchors, 4]

    # these proposals need filtering
    all_proposals = {
        'boxes': boxes,
        'probabilities': probabilities,
    }

    # this will be used for computing the loss
    rpn_output = {
        'encoded_boxes': encoded_boxes,
        'objectness_scores': objectness_scores
    }

    return all_proposals, rpn_output, anchors


def choose_proposals(all_proposals, params):
    """
    Returns:
        rois: a list of float tensors,
            where `i`-th tensor has shape [num_proposals_i, 4],
            and `sum_i num_proposals_i = total_num_proposals`.
        roi_image_indices: an int tensor with shape [total_num_proposals].
        num_proposals_per_image: an int tensor with shape [batch_size].
    """

    batch_size = all_proposals['boxes'].shape[0].value
    boxes_per_image = tf.unstack(all_proposals['boxes'], axis=0)
    probabilities_per_image = tf.unstack(all_proposals['probabilities'], axis=0)

    rois, roi_image_indices, num_proposals = [], [], []
    for n, b, p in zip(range(batch_size), boxes_per_image, probabilities_per_image):
        # `n` - index of an image in the batch

        # let's hope boxes in b have strictly positive area
        to_keep = tf.image.non_max_suppression(
            b, p, params['nms_max_output_size'],
            params['proposal_iou_threshold'],
            params['before_nms_score_threshold']
        )
        b = tf.gather(b, to_keep)
        k = tf.size(to_keep)

        # uncomment this to do "approximate joint training"
        # b = tf.stop_gradient(b)
        # k = tf.stop_gradient(k)

        rois.append(b)
        roi_image_indices.append(tf.fill([k], n))
        num_proposals.append(k)

    proposals = {
        'rois': rois,
        'roi_image_indices': tf.concat(roi_image_indices, axis=0),
        'num_proposals_per_image': tf.stack(num_proposals, axis=0)
    }
    return proposals
