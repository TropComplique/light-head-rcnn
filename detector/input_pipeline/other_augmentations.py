import tensorflow as tf

"""
There are various data augmentations for training object detectors.

`image` is assumed to be a float tensor with shape [height, width, 3],
it is a RGB image with pixel values in range [0, 1].
And box coordinates are normalized.
"""


def random_color_manipulations(image, probability=0.1, grayscale_probability=0.1):

    def manipulate(image):
        br_delta = tf.random_uniform([], -32.0/255.0, 32.0/255.0)
        cb_factor = tf.random_uniform([], -0.1, 0.1)
        cr_factor = tf.random_uniform([], -0.1, 0.1)
        channels = tf.split(axis=2, num_or_size_splits=3, value=image)
        red_offset = 1.402 * cr_factor + br_delta
        green_offset = -0.344136 * cb_factor - 0.714136 * cr_factor + br_delta
        blue_offset = 1.772 * cb_factor + br_delta
        channels[0] += red_offset
        channels[1] += green_offset
        channels[2] += blue_offset
        image = tf.concat(axis=2, values=channels)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def to_grayscale(image):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        return image

    with tf.name_scope('random_color_manipulations'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: manipulate(image), lambda: image)

    with tf.name_scope('to_grayscale'):
        do_it = tf.less(tf.random_uniform([]), grayscale_probability)
        image = tf.cond(do_it, lambda: to_grayscale(image), lambda: image)

    return image


def random_flip_left_right(image, boxes):

    def flip(image, boxes):
        flipped_image = tf.image.flip_left_right(image)
        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        flipped_xmin = tf.subtract(1.0, xmax)
        flipped_xmax = tf.subtract(1.0, xmin)
        flipped_boxes = tf.stack([ymin, flipped_xmin, ymax, flipped_xmax], 1)
        return flipped_image, flipped_boxes

    with tf.name_scope('random_flip_left_right'):
        do_it = tf.less(tf.random_uniform([]), 0.5)
        image, boxes = tf.cond(do_it, lambda: flip(image, boxes), lambda: (image, boxes))
        return image, boxes


def random_pixel_value_scale(image, minval=0.9, maxval=1.1, probability=0.1):
    """This function scales each pixel independently of the other ones.

    Arguments:
        image: a float tensor with shape [height, width, 3],
            an image with pixel values varying between [0, 1].
        minval: a float number, lower ratio of scaling pixel values.
        maxval: a float number, upper ratio of scaling pixel values.
        probability: a float number.
    Returns:
        a float tensor with shape [height, width, 3].
    """
    def random_value_scale(image):
        color_coefficient = tf.random_uniform(
            tf.shape(image), minval=minval,
            maxval=maxval, dtype=tf.float32
        )
        image = tf.multiply(image, color_coefficient)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    with tf.name_scope('random_pixel_value_scale'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: random_value_scale(image), lambda: image)
        return image


def random_jitter_boxes(boxes, ratio=0.05):
    """Randomly jitter bounding boxes.

    Arguments:
        boxes: a float tensor with shape [N, 4].
        ratio: a float number.
            The ratio of the box width and height that the corners can jitter.
            For example if the width is 100 pixels and ratio is 0.05,
            the corners can jitter up to 5 pixels in the x direction.
    Returns:
        a float tensor with shape [N, 4].
    """
    def random_jitter_box(box, ratio):
        """Randomly jitter a box.
        Arguments:
            box: a float tensor with shape [4].
            ratio: a float number.
        Returns:
            a float tensor with shape [4].
        """
        ymin, xmin, ymax, xmax = tf.unstack(box, axis=0)
        box_height, box_width = ymax - ymin, xmax - xmin
        hw_coefs = tf.stack([box_height, box_width, box_height, box_width])

        rand_numbers = tf.random_uniform(
            [4], minval=-ratio, maxval=ratio, dtype=tf.float32
        )
        hw_rand_coefs = tf.multiply(hw_coefs, rand_numbers)

        jittered_box = tf.add(box, hw_rand_coefs)
        return jittered_box

    with tf.name_scope('random_jitter_boxes'):
        distorted_boxes = tf.map_fn(
            lambda x: random_jitter_box(x, ratio),
            boxes, dtype=tf.float32, back_prop=False
        )
        distorted_boxes = tf.clip_by_value(distorted_boxes, 0.0, 1.0)
        return distorted_boxes
