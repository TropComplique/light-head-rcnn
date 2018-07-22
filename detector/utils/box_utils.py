import tensorflow as tf
from detector.constants import EPSILON, SCALE_FACTORS


"""
Tools for dealing with bounding boxes.

All boxes are of the format [ymin, xmin, ymax, xmax] if not stated otherwise.
Also the following must be true: ymin < ymax and xmin < xmax.
And all box coordinates are absolute.
"""


def iou(boxes1, boxes2):
    """Computes pairwise intersection-over-union between two box collections.
    Arguments:
        boxes1: a float tensor with shape [N, 4].
        boxes2: a float tensor with shape [M, 4].
    Returns:
        a float tensor with shape [N, M] representing pairwise iou scores.
    """
    with tf.name_scope('iou'):
        intersections = intersection(boxes1, boxes2)
        areas1 = area(boxes1)
        areas2 = area(boxes2)
        unions = tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections
        return tf.clip_by_value(tf.divide(intersections, unions + EPSILON), 0.0, 1.0)


def intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.
    Arguments:
        boxes1: a float tensor with shape [N, 4].
        boxes2: a float tensor with shape [M, 4].
    Returns:
        a float tensor with shape [N, M] representing pairwise intersections.
    """
    ymin1, xmin1, ymax1, xmax1 = tf.split(boxes1, num_or_size_splits=4, axis=1)
    ymin2, xmin2, ymax2, xmax2 = tf.split(boxes2, num_or_size_splits=4, axis=1)
    # they all have shapes like [None, 1]

    all_pairs_min_ymax = tf.minimum(ymax1, tf.transpose(ymax2))
    all_pairs_max_ymin = tf.maximum(ymin1, tf.transpose(ymin2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(xmax1, tf.transpose(xmax2))
    all_pairs_max_xmin = tf.maximum(xmin1, tf.transpose(xmin2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    # they all have shape [N, M]

    return intersect_heights * intersect_widths


def area(boxes):
    """Computes area of boxes.
    Arguments:
        boxes: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N] representing box areas.
    """
    ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
    return (ymax - ymin) * (xmax - xmin)


def to_center_coordinates(boxes):
    """Convert bounding boxes of the format
    [ymin, xmin, ymax, xmax] to the format [cy, cx, h, w].

    Arguments:
        boxes: a float tensor with shape [N, 4].
    Returns:
        a list of float tensors with shape [N]
        that represent cy, cx, h, w.
    """
    ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
    h, w = ymax - ymin, xmax - xmin
    cy, cx = ymin + 0.5*h, xmin + 0.5*w
    return [cy, cx, h, w]


def encode(boxes, anchors):
    """Encode boxes with respect to anchors (or proposals).

    Arguments:
        boxes: a float tensor with shape [N, 4].
        anchors: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N, 4],
        anchor-encoded boxes of the format [ty, tx, th, tw].
    """
    with tf.name_scope('encode_groundtruth'):

        ycenter_a, xcenter_a, ha, wa = to_center_coordinates(anchors)
        ycenter, xcenter, h, w = to_center_coordinates(boxes)

        # to avoid NaN in division and log below
        ha += EPSILON
        wa += EPSILON
        h += EPSILON
        w += EPSILON

        tx = (xcenter - xcenter_a)/wa
        ty = (ycenter - ycenter_a)/ha
        tw = tf.log(w / wa)
        th = tf.log(h / ha)

        ty *= SCALE_FACTORS[0]
        tx *= SCALE_FACTORS[1]
        th *= SCALE_FACTORS[2]
        tw *= SCALE_FACTORS[3]

        return tf.stack([ty, tx, th, tw], axis=1)


def decode(codes, anchors):
    """Decode relative codes to normal boxes.

    Arguments:
        codes: a float tensor with shape [N, 4],
            anchor-encoded boxes of the format [ty, tx, th, tw].
        anchors: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N, 4],
        bounding boxes of the format [ymin, xmin, ymax, xmax].
    """
    with tf.name_scope('decode_predictions'):

        ycenter_a, xcenter_a, ha, wa = to_center_coordinates(anchors)
        ty, tx, th, tw = tf.unstack(codes, axis=1)

        ty /= SCALE_FACTORS[0]
        tx /= SCALE_FACTORS[1]
        th /= SCALE_FACTORS[2]
        tw /= SCALE_FACTORS[3]

        w = tf.exp(tw) * wa
        h = tf.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a

        ymin, xmin = ycenter - 0.5*h, xcenter - 0.5*w
        ymax, xmax = ycenter + 0.5*h, xcenter + 0.5*w
        return tf.stack([ymin, xmin, ymax, xmax], axis=1)


def batch_decode(box_encodings, anchors):
    """Decodes a batch of box encodings with respect to anchors (or proposals).

    Arguments:
        box_encodings: a float tensor with shape [batch_size, num_anchors, 4].
        anchors: a float tensor with shape [num_anchors, 4].
    Returns:
        a float tensor with shape [batch_size, num_anchors, 4].
    """
    with tf.name_scope('batch_decode'):

        # batch size is a static value (it must be known when building the graph)
        batch_size = box_encodings.shape[0].value

        # number of anchors depends on the image size (it is a dynamic value)
        num_anchors = tf.shape(anchors)[0]

        tiled_anchor_boxes = tf.tile(
            tf.expand_dims(anchors, 0),
            [batch_size, 1, 1]
        )  # shape [batch_size, num_anchors, 4]
        decoded_boxes = decode(
            tf.reshape(box_encodings, [-1, 4]),
            tf.reshape(tiled_anchor_boxes, [-1, 4])
        )  # shape [batch_size * num_anchors, 4]

        decoded_boxes = tf.reshape(decoded_boxes, [batch_size, num_anchors, 4])
        return decoded_boxes


def prune_outside_window(boxes, window):
    """
    Arguments:
        boxes: a float tensor with shape [N, 4].
        window: a float tensor with shape [4]
            representing [ymin, xmin, ymax, xmax] of the window.
    Returns:
        pruned_boxes: a float tensor with shape [M, 4], where M <= N.
        valid_indices: an int tensor with shape [M] indexing
            the valid bounding boxes in the input tensor.
    """
    with tf.name_scope('prune_outside_window'):
        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        win_ymin, win_xmin, win_ymax, win_xmax = tf.unstack(window, axis=0)
        coordinate_violations = tf.stack([
            tf.less(ymin, win_ymin), tf.less(xmin, win_xmin),
            tf.greater(ymax, win_ymax), tf.greater(xmax, win_xmax)
        ], axis=1)
        valid_indices = tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, axis=1)))
        valid_indices = tf.squeeze(valid_indices, axis=1)
        pruned_boxes = tf.gather(boxes, valid_indices)
        return pruned_boxes, valid_indices


def clip_by_window(boxes, window):
    """
    Arguments:
        boxes: a float tensor with shape [N, 4].
        window: a float tensor with shape [4]
            representing [ymin, xmin, ymax, xmax] of the window.
    Returns:
        a float tensor with shape [N, 4].
    """
    with tf.name_scope('clip_by_window'):
        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        win_ymin, win_xmin, win_ymax, win_xmax = tf.unstack(window, axis=0)
        ymin = tf.clip_by_value(ymin, win_ymin, win_ymax)
        xmin = tf.clip_by_value(xmin, win_xmin, win_xmax)
        ymax = tf.clip_by_value(ymax, win_ymin, win_ymax)
        xmax = tf.clip_by_value(xmax, win_xmin, win_xmax)
        return tf.stack([ymin, xmin, ymax, xmax], axis=1)
