
wider_params = {

    # TRAINING PARAMETERS:

    'model_dir': '/home/dan/work/light-head-rcnn/models/run00/',
    'train_dataset_path': '/mnt/datasets/dan/wider_train_shards/',
    'pretrained_checkpoint': 'pretrained/resnet_v1_50.ckpt',

    'num_classes': 1,
    'labels': '/home/dan/work/light-head-rcnn/data/wider_labels.txt',

    # a list of integers
    'training_min_dimensions': [600, 700, 800, 900, 1000],
    # an integer or None
    'training_max_dimension': 1400,

    'num_steps': 300000,
    'lr_boundaries': [160000, 200000],
    'lr_values': [0.0004, 0.00004, 0.000004],

    'min_proposal_area': 64,
    'before_nms_score_threshold': 0.01,
    # an integer
    'nms_max_output_size': 300,
    # a float number
    'proposal_iou_threshold': 0.7,

    'alpha': 1.0,
    'beta': 1.0,
    'gamma': 1.0,
    'theta': 1.0,
    
    'positives_threshold': 0.7,
    'negatives_threshold': 0.3,
    'second_stage_threshold': 0.5,
    'first_stage_batch_size': 256,
    'positive_fraction': 0.5,
    'num_hard_examples': 256,
    
    'weight_decay': 1e-5,

    # EVALUATION PARAMETERS:

    'val_dataset_path': '/mnt/datasets/dan/wider_val_shards/',

    # an integer or None
    'evaluation_min_dimension': 800,
    # an integer or None
    'evaluation_max_dimension': 1200,

    'score_threshold': 0.1,
    'iou_threshold': 0.4,
    'max_boxes_per_class': 100,

    # MODEL PARAMETERS:

    # an integer
    'p': 7,
    # an integer
    'k': 15,
    # an integer
    'channels_middle': 256,

    # a tuple of integers (crop_height, crop_width)
    'crop_size': (14, 14),
    # a tuple of integers (spatial_bins_y, spatial_bins_x)
    'num_spatial_bins': (7, 7),

    # a list of integers
    'scales': [16, 32, 64, 128, 256, 512],
    # a list of float numbers
    'aspect_ratios': [0.5, 1.0, 2.0],
    # a tuple of integers
    'anchor_stride': (16, 16),
    # a tuple of integers
    'anchor_offset': (0, 0),

}
