
wider_params = {

    'model_dir': '/home/dan/work/light-head-rcnn/models/run00/',
    'train_dataset_path': '/mnt/datasets/dan/wider_train_shards/',
    'val_dataset_path': '/mnt/datasets/dan/wider_val_shards/',
    'pretrained_checkpoint': 'pretrained/resnet_v1_50.ckpt',
    'labels': '/home/dan/work/light-head-rcnn/data/wider_labels.txt',
    'num_classes': 1,

    # IMAGE SIZES:

    # an integer or None
    'evaluation_min_dimension': 800,
    # an integer or None
    'evaluation_max_dimension': 1200,

    # a list of integers
    'training_min_dimensions': [400, 600, 700, 800, 1000, 1200],
    # an integer or None
    'training_max_dimension': 1400,

    'num_steps': 500000,
    'lr_boundaries': [300000, 450000],
    'lr_values': [6e-4, 6e-5, 6e-6],

    # PROPOSAL GENERATION:

    # an integer
    'min_proposal_area': 64,
    # a float number
    'before_nms_score_threshold': 0.01,
    # an integer
    'nms_max_output_size': 1000,
    # a float number
    'proposal_iou_threshold': 0.7,

    # LOSS:

    # float numbers, weights for losses
    'alpha': 1.0,  # rpn localization
    'beta': 2.0,  # rpn classification
    'gamma': 1.0,  # roi localization
    'theta': 2.0,  # roi classification

    'first_stage_batch_size': 256,
    'positive_fraction': 0.5,
    'num_hard_examples': 256,

    # BOX MATCHING:

    'positives_threshold': 0.5,  # or 0.7
    'negatives_threshold': 0.3,
    'second_stage_threshold': 0.5,

    # FINAL POSTPROCESSING:

    'score_threshold': 0.1,
    'iou_threshold': 0.4,
    'max_boxes_per_class': 100,

    # FEATURE EXTRACTOR:

    'backbone': 'resnet',  # 'resnet' or 'mobilenet'
    # a float number, relevant only for mobilenet
    'depth_multiplier': 1.0,
    'weight_decay': 1e-4,

    # THE HEAD:

    # an integer
    'p': 7,
    # an integer
    'k': 15,
    # an integer
    'channels_middle': 256,

    # PS ROI ALIGN POOLING:

    # a tuple of integers (crop_height, crop_width)
    'crop_size': (14, 14),
    # a tuple of integers (spatial_bins_y, spatial_bins_x)
    'num_spatial_bins': (7, 7),

    # ANCHOR GENERATION:

    # a list of integers
    'scales': [16, 32, 64, 128, 256, 512],
    # a list of float numbers
    'aspect_ratios': [0.5, 1.0, 2.0],
    # a tuple of integers
    'anchor_stride': (16, 16),
    # a tuple of integers
    'anchor_offset': (0, 0),
}

wider_light_params = wider_params.copy()
wider_light_params.update({
    'model_dir': '/home/dan/work/light-head-rcnn/models/run01/',
    'pretrained_checkpoint': 'pretrained/mobilenet_v2_0.5_224.ckpt',
    'backbone': 'mobilenet',
    'depth_multiplier': 0.5,
    'channels_middle': 64,
    'scales': [32, 64, 128, 256, 512],
})

coco_params = wider_params.copy()
coco_params.update({
    'model_dir': '/home/dan/work/light-head-rcnn/models/run02/',
    'train_dataset_path': '/mnt/datasets/dan/coco_train_shards/',
    'val_dataset_path': '/mnt/datasets/dan/coco_val_shards/',
    'pretrained_checkpoint': 'pretrained/mobilenet_v2_0.5_224.ckpt',
    'labels': '/home/dan/work/light-head-rcnn/data/coco_labels.txt',
    'num_classes': 80,
    'positives_threshold': 0.7,
    'backbone': 'mobilenet',
    'depth_multiplier': 0.5,
    'weight_decay': 1e-4,
    'channels_middle': 64,
    'scales': [32, 64, 128, 256, 512],
})
