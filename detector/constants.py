import tensorflow as tf

WEIGHT_DECAY = 1e-4
BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-5

ANCHOR_STRIDE = 16
SCALES = [32, 64, 128, 256, 512]
ASPECT_RATIOS = [0.5, 1.0, 2.0]

RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR
SHUFFLE_BUFFER_SIZE = 2000
NUM_PARALLEL_CALLS = 8

EPSILON = 1e-8

# this is used when we are doing box encoding/decoding
SCALE_FACTORS = [10.0, 10.0, 5.0, 5.0]
