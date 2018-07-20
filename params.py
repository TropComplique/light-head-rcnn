
params = {
    'num_classes': 1,

    # head
    'p': 7, # an integer.
    'k': 15,  # an integer.
    'channels_middle': 256, # an integer.
    'crop_size': ,
    'num_spatial_bins':,

    'scales': [16, 32, 64, 128, 256, 512],
    'aspect_ratios': [0.5, 1.0, 2.0],
    'anchor_stride': (16, 16),
    'anchor_offset': (0, 0),

    'before_nms_top_k': ,
    'nms_max_output_size': ,
    'iou_threshold': ,

    # a list of integers
    'training_min_dimensions': [],
    # an int or None
    'training_max_dimension': ,
    'evaluation_min_dimension': ,  # or None
    'evaluation_max_dimension': ,
}
