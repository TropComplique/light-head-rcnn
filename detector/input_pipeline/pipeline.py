import tensorflow as tf
from detector.constants import SHUFFLE_BUFFER_SIZE, NUM_PARALLEL_CALLS, RESIZE_METHOD
from .random_image_crop import random_image_crop
from .other_augmentations import random_color_manipulations,\
    random_flip_left_right, random_pixel_value_scale, random_jitter_boxes


class Pipeline:
    """
    Input pipeline for training or evaluation of object detectors.
    Note that it only outputs batches of size 1.
    """

    def __init__(self, filenames, is_training, params):
        """
        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            is_training: a boolean.
            params: a dict.
        """
        self.is_training = is_training
        self.params = params

        def get_num_samples(filename):
            return sum(1 for _ in tf.python_io.tf_record_iterator(filename))

        num_examples = 0
        for filename in filenames:
            num_examples_in_file = get_num_samples(filename)
            assert num_examples_in_file > 0
            num_examples += num_examples_in_file
        assert num_examples > 0

        # because datasets are big
        # we split them into many small pieces (shards)
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        num_shards = len(filenames)

        if is_training:
            dataset = dataset.shuffle(buffer_size=num_shards)

        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.prefetch(buffer_size=1)

        if is_training:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

        dataset = dataset.repeat(None if is_training else 1)
        dataset = dataset.map(
            self.parse_and_preprocess,
            num_parallel_calls=NUM_PARALLEL_CALLS
        )
        # we want batches with static first dimension:
        dataset = dataset.batch(batch_size=1, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=1)

        self.dataset = dataset

    def parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. Possibly augments it.

        Returns:
            image: a float tensor with shape [height, width, 3],
                an RGB image with pixel values in the range [0, 1].
            boxes: a float tensor with shape [num_boxes, 4],
                box coordinates are absolute.
            labels: an int tensor with shape [num_boxes].
            num_boxes: an int tensor with shape [].
        """
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'ymin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'ymax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'labels': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get an image
        image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # now pixel values are scaled to [0, 1] range

        # get labels
        labels = tf.to_int32(parsed_features['labels'])

        # get groundtruth boxes, they must be in from-zero-to-one format,
        # also, it assumed that ymin < ymax and xmin < xmax
        boxes = tf.stack([
            parsed_features['ymin'], parsed_features['xmin'],
            parsed_features['ymax'], parsed_features['xmax']
        ], axis=1)
        boxes = tf.to_float(boxes)
        # box coordinates are relative (normalized) here

        if self.is_training:
            image, boxes, labels = self.augmentation(image, boxes, labels)
        else:
            min_dimension = self.params['evaluation_min_dimension']
            max_dimension = self.params['evaluation_max_dimension']
            if min_dimension is not None:
                image = resize_keeping_aspect_ratio(image, min_dimension, max_dimension)

        # convert to absolute coordinates
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        scaler = tf.to_float(tf.stack([height, width, height, width]))
        boxes = boxes * scaler

        # in the format required by tf.estimator,
        # they will be batched later
        features = {'images': image}
        labels = {'boxes': boxes, 'labels': labels, 'num_boxes': tf.to_int32(tf.shape(boxes)[0])}
        return features, labels

    def augmentation(self, image, boxes, labels):
        # there are a lot of hyperparameters here,
        # you will need to tune them all, haha

        image, boxes, labels = random_image_crop(
            image, boxes, labels, probability=0.2,
            min_object_covered=0.9,
            aspect_ratio_range=(0.75, 1.33),
            area_range=(0.4, 0.8),
            overlap_thresh=0.3
        )

        image = random_color_manipulations(image, probability=0.2, grayscale_probability=0.05)
        image = random_pixel_value_scale(image, minval=0.8, maxval=1.2, probability=0.1)
        boxes = random_jitter_boxes(boxes, ratio=0.01)
        image, boxes = random_flip_left_right(image, boxes)

        def random_resize(image):
            """Multiscale training implementation."""

            training_min_dimensions = self.params['training_min_dimensions']
            training_max_dimension = self.params['training_max_dimension']

            # choose a random minimal dimension
            min_dimensions = tf.constant(training_min_dimensions, dtype=tf.int32)
            random_min_dimension = tf.random_shuffle(min_dimensions)[0]

            image = resize_keeping_aspect_ratio(
                image, random_min_dimension,
                training_max_dimension
            )
            return image

        image = random_resize(image)
        return image, boxes, labels


def resize_keeping_aspect_ratio(image, min_dimension, max_dimension=None):
    """
    The output size can be described by two cases:
    1. If the image can be rescaled so its minimum dimension is equal to the
       provided value without the other dimension exceeding max_dimension,
       then do so.
    2. Otherwise, resize so the largest dimension is equal to max_dimension.

    Arguments:
        image: a float tensor with shape [height, width, 3].
        min_dimension: an int tensor with shape [].
        max_dimension: an int tensor with shape [] or None.
    Returns:
        a float tensor with shape [new_height, new_width, 3].
    """
    image_shape = tf.shape(image)
    height = tf.to_float(image_shape[0])
    width = tf.to_float(image_shape[1])

    original_min_dim = tf.minimum(height, width)
    min_dimension = tf.to_float(min_dimension)
    scale_factor1 = min_dimension / original_min_dim
    height1 = tf.round(height * scale_factor1)
    width1 = tf.round(width * scale_factor1)

    new_height, new_width = height1, width1

    if max_dimension is not None:

        original_max_dim = tf.maximum(height, width)
        max_dimension = tf.to_float(max_dimension)
        scale_factor2 = max_dimension / original_max_dim
        height2 = tf.round(height * scale_factor2)
        width2 = tf.round(width * scale_factor2)

        new_height, new_width = tf.cond(
            tf.maximum(height1, width1) < max_dimension,
            lambda: (height1, width1),
            lambda: (height2, width2)
        )

    new_size = [tf.to_int32(new_height), tf.to_int32(new_width)]
    image = tf.image.resize_images(image, new_size, method=RESIZE_METHOD)
    return image
