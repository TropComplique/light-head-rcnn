import tensorflow as tf


"""
It is based on function `position_sensitive_crop_regions` from here:
https://github.com/tensorflow/models/blob/master/research/object_detection/utils/ops.py
"""


def position_sensitive_roi_align_pooling(
        features, boxes, box_image_indices,
        crop_size=(14, 14), num_spatial_bins=(7, 7)):
    """
    Arguments:
        features: a float tensor with shape [batch, height, width, depth].
        boxes: a float tensor with shape [num_boxes, 4]. The i-th row of the tensor
            specifies the coordinates of a box in the `box_image_indices[i]` image
            and is specified in normalized coordinates [ymin, xmin, ymax, xmax].
        box_image_indices: an int tensor with shape [num_boxes]. It has values in range [0, batch).
        crop_size: a tuple with two integers (crop_height, crop_width).
        num_spatial_bins: a tuple with two integers (spatial_bins_y, spatial_bins_x).
            Represents the number of position-sensitive bins in y and x directions.
            Both values should be >= 1. `crop_height` should be
            divisible by `spatial_bins_y`, and similarly for width.
            The number of `features` channels should be divisible by (spatial_bins_y * spatial_bins_x).
    Returns:
        a float tensor with shape [num_boxes, spatial_bins_y * spatial_bins_x, crop_channels],
            where `crop_channels = depth/(spatial_bins_y * spatial_bins_x)`.
    """
    total_bins = 1
    bin_crop_size = []

    for num_bins, crop_dim in zip(num_spatial_bins, crop_size):
        assert num_bins >= 1
        assert crop_dim % num_bins == 0
        total_bins *= num_bins
        bin_crop_size.append(crop_dim // num_bins)

    depth = features.shape[3].value
    assert depth % total_bins == 0

    ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
    spatial_bins_y, spatial_bins_x = num_spatial_bins
    step_y = (ymax - ymin) / spatial_bins_y
    step_x = (xmax - xmin) / spatial_bins_x

    # split each box into `total_bins` bins
    position_sensitive_boxes = []
    for bin_y in range(spatial_bins_y):
        for bin_x in range(spatial_bins_x):
            box_coordinates = [
                ymin + bin_y * step_y,
                xmin + bin_x * step_x,
                ymin + (bin_y + 1) * step_y,
                xmin + (bin_x + 1) * step_x,
            ]
            position_sensitive_boxes.append(tf.stack(box_coordinates, axis=1))

    feature_splits = tf.split(features, num_or_size_splits=total_bins, axis=3)
    # it a list of float tensors with
    # shape [batch_size, image_height, image_width, depth/total_bins]
    # and it has length `total_bins`

    feature_crops = []
    for split, box in zip(feature_splits, position_sensitive_boxes):
        crop = tf.image.crop_and_resize(
            split, box, box_image_indices,
            bin_crop_size, method='bilinear'
        )
        # shape [num_boxes, crop_height/spatial_bins_y, crop_width/spatial_bins_x, depth/total_bins]

        # do max pooling over spatial positions within the bin
        crop = tf.reduce_max(crop, axis=[1, 2])
        crop = tf.expand_dims(crop, 1)
        # shape [num_boxes, 1, depth/total_bins]

        feature_crops.append(crop)

    return tf.concat(feature_crops, axis=1)
