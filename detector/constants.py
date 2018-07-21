import tensorflow as tf

EPSILON = 1e-8

BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-5

RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR
SHUFFLE_BUFFER_SIZE = 5000
NUM_PARALLEL_CALLS = 8

# this is used when we are doing box encoding/decoding
SCALE_FACTORS = [10.0, 10.0, 5.0, 5.0]
