import tensorflow as tf
from detector.utils import area, intersection
from detector.constants import EPSILON


"""Here it is assumed that box coordinates are normalized."""


def random_image_crop(
        image, boxes, labels, probability=0.1,
        min_object_covered=0.9, aspect_ratio_range=(0.75, 1.33),
        area_range=(0.5, 1.0), overlap_thresh=0.3):

    def crop(image, boxes, labels):
        image, boxes, keep_indices = randomly_crop_image(
            image, boxes, min_object_covered,
            aspect_ratio_range,
            area_range, overlap_thresh
        )
        labels = tf.gather(labels, keep_indices)
        return image, boxes, labels

    do_it = tf.less(tf.random_uniform([]), probability)
    image, boxes, labels = tf.cond(
        do_it,
        lambda: crop(image, boxes, labels),
        lambda: (image, boxes, labels)
    )
    return image, boxes, labels


def randomly_crop_image(
        image, boxes, min_object_covered=0.9,
        aspect_ratio_range=(0.75, 1.33), area_range=(0.5, 1.0),
        overlap_thresh=0.3):
    """Performs random crop. Given the input image and its bounding boxes,
    this op randomly crops a subimage.  Given a user-provided set of input constraints,
    the crop window is resampled until it satisfies these constraints.
    If within 100 trials it is unable to find a valid crop, the original
    image is returned. Both input boxes and returned boxes are in normalized
    form (e.g., lie in the unit square [0, 1]).

    Arguments:
        image: a float tensor with shape [height, width, 3],
            with pixel values varying between [0, 1].
        boxes: a float tensor containing bounding boxes. It has shape
            [num_boxes, 4]. Boxes are in normalized form, meaning
            their coordinates vary between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].
        min_object_covered: the cropped image must cover at least this fraction of
            at least one of the input bounding boxes.
        aspect_ratio_range: allowed range for aspect ratio of cropped image.
        area_range: allowed range for area ratio between cropped image and the
            original image.
        overlap_thresh: minimum overlap thresh with new cropped
            image to keep the box.
    Returns:
        image: cropped image.
        boxes: remaining boxes.
        keep_indices: indices of remaining boxes in input boxes tensor.
            They are used to get a slice from the 'labels' tensor.
            len(keep_indices) = len(boxes).
    """
    with tf.name_scope('random_crop_image'):

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(boxes, 0),
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=100,
            use_image_if_no_bounding_boxes=True
        )
        begin, size, window = sample_distorted_bounding_box
        image = tf.slice(image, begin, size)
        image.set_shape([None, None, 3])
        window = tf.squeeze(window, axis=[0, 1])

        # remove boxes that are completely outside the cropped image
        boxes, inside_window_ids = prune_completely_outside_window(boxes, window)

        # remove boxes that are too much outside the cropped image
        boxes, keep_indices = prune_non_overlapping_boxes(
            boxes, tf.expand_dims(window, 0),
            min_overlap=overlap_thresh
        )

        # change coordinates of the remaining boxes
        boxes = change_coordinate_frame(boxes, window)

        keep_indices = tf.gather(inside_window_ids, keep_indices)
        return image, boxes, keep_indices


def prune_completely_outside_window(boxes, window):
    """Prunes bounding boxes that fall completely outside of the given window.
    This function does not clip partially overflowing boxes.
    Arguments:
        boxes: a float tensor with shape [M_in, 4].
        window: a float tensor with shape [4] representing [ymin, xmin, ymax, xmax]
            of the window.
    Returns:
        boxes: a float tensor with shape [M_out, 4] where 0 <= M_out <= M_in.
        valid_indices: a long tensor with shape [M_out] indexing the valid bounding boxes
            in the input 'boxes' tensor.
    """
    with tf.name_scope('prune_completely_outside_window'):

        y_min, x_min, y_max, x_max = tf.split(boxes, num_or_size_splits=4, axis=1)
        # they have shape [None, 1]
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        # they have shape []

        coordinate_violations = tf.concat([
            tf.greater_equal(y_min, win_y_max), tf.greater_equal(x_min, win_x_max),
            tf.less_equal(y_max, win_y_min), tf.less_equal(x_max, win_x_min)
        ], axis=1)
        valid_indices = tf.squeeze(
            tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))),
            axis=1
        )
        boxes = tf.gather(boxes, valid_indices)
        return boxes, valid_indices


def prune_non_overlapping_boxes(boxes1, boxes2, min_overlap):
    """Prunes the boxes in boxes1 that overlap less than thresh with boxes2.
    For each box in boxes1, we want its IOA to be more than min_overlap with
    at least one of the boxes in boxes2. If it does not, we remove it.

    Arguments:
        boxes1: a float tensor with shape [N, 4].
        boxes2: a float tensor with shape [M, 4].
        min_overlap: minimum required overlap between boxes,
            to count them as overlapping.
    Returns:
        boxes: a float tensor with shape [N', 4].
        keep_indices: a long tensor with shape [N'] indexing kept bounding boxes in the
            first input tensor ('boxes1').
    """
    with tf.name_scope('prune_non_overlapping_boxes'):

        overlap = ioa(boxes2, boxes1)  # shape [M, N]
        overlap = tf.reduce_max(overlap, axis=0)  # shape [N]

        keep_bool = tf.greater_equal(overlap, min_overlap)
        keep_indices = tf.squeeze(tf.where(keep_bool), axis=1)

        boxes = tf.gather(boxes1, keep_indices)
        return boxes, keep_indices


def change_coordinate_frame(boxes, window):
    """Change coordinate frame of the boxes to be relative to window's frame.

    Arguments:
        boxes: a float tensor with shape [N, 4].
        window: a float tensor with shape [4].
    Returns:
        a float tensor with shape [N, 4].
    """
    with tf.name_scope('change_coordinate_frame'):

        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        ymin -= window[0]
        xmin -= window[1]
        ymax -= window[0]
        xmax -= window[1]

        win_height = window[2] - window[0]
        win_width = window[3] - window[1]
        boxes = tf.stack([
            ymin/win_height, xmin/win_width,
            ymax/win_height, xmax/win_width
        ], axis=1)
        boxes = tf.clip_by_value(boxes, 0.0, 1.0)
        return boxes


def ioa(boxes1, boxes2):
    """Computes pairwise intersection-over-area between box collections.
    intersection-over-area (IOA) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, ioa(box1, box2) != ioa(box2, box1).

    Arguments:
        boxes1: a float tensor with shape [N, 4].
        boxes2: a float tensor with shape [M, 4].
    Returns:
        a float tensor with shape [N, M] representing pairwise ioa scores.
    """
    with tf.name_scope('ioa'):
        intersections = intersection(boxes1, boxes2)  # shape [N, M]
        areas = tf.expand_dims(area(boxes2), 0)  # shape [1, M]
        return tf.clip_by_value(tf.divide(intersections, areas + EPSILON), 0.0, 1.0)
