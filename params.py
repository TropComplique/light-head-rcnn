
wider_params = {

    'model_dir': '/home/dan/work/light-head-rcnn/models/run00/',
    'train_dataset_path': '/mnt/datasets/dan/wider_train_shards/',
    'val_dataset_path': '/mnt/datasets/dan/wider_val_shards/',
    'pretrained_checkpoint': 'pretrained/resnet_v1_50.ckpt',
    'num_classes': 1,

    # IMAGE SIZES:

    # an integer or None
    'evaluation_min_dimension': 800,
    # an integer or None
    'evaluation_max_dimension': 1200,

    # a list of integers
    'training_min_dimensions': [600, 700, 800, 1000, 1200],
    # an integer or None
    'training_max_dimension': 1400,

    'num_steps': 400000,
    'lr_boundaries': [250000, 350000],
    'lr_values': [1e-3, 1e-4, 1e-5],

    # PROPOSAL GENERATION:

    # an integer
    'min_proposal_area': 36,
    # a float number
    'before_nms_score_threshold': 1e-6,
    # an integer
    'nms_max_output_size': 1000,
    # a float number
    'proposal_iou_threshold': 0.7,
    # an integer
    'rpn_num_channels': 128,

    # LOSS:

    # float numbers, weights for losses
    'alpha': 1.0,  # rpn localization
    'beta': 1.0,  # rpn classification
    'gamma': 2.0,  # roi localization
    'theta': 1.0,  # roi classification

    'first_stage_batch_size': 256,
    'positive_fraction': 0.5,
    'num_hard_examples': 256,

    # BOX MATCHING:

    'positives_threshold': 0.5,
    'negatives_threshold': 0.3,
    'second_stage_threshold': 0.5,

    # FINAL POSTPROCESSING:

    'score_threshold': 0.01,
    'iou_threshold': 0.4,
    'max_boxes_per_class': 150,

    # FEATURE EXTRACTOR:

    'backbone': 'resnet',  # 'resnet' or 'mobilenet' or 'shufflenet'
    'depth_multiplier': 1.0,  # a float number, relevant only for mobilenet and shufflenet
    'weight_decay': 1e-4,

    # THE HEAD:

    # an integer, ps roi align pooling
    'p': 7,
    # an integer
    'k': 15,
    # an integer
    'channels_middle': 64,
    'fc_layer_size': 512,

    # ANCHOR GENERATION:

    # a list of integers
    'scales': [16, 32, 64, 128, 256, 512],
    # a list of float numbers
    'aspect_ratios': [1.0],
    # a tuple of integers
    'anchor_stride': (16, 16),
    # a tuple of integers
    'anchor_offset': (0, 0),
}

wider_light_params = wider_params.copy()
wider_light_params.update({
    'model_dir': '/home/dan/work/light-head-rcnn/models/run01/',
    'pretrained_checkpoint': 'pretrained/shufflenet_v2_1.0x/model.ckpt-1661328',
    'backbone': 'shufflenet',
    'depth_multiplier': 1.0,
    'channels_middle': 48,
    'rpn_num_channels': 128,
    'fc_layer_size': 256,
})

coco_params = wider_params.copy()
coco_params.update({
    'model_dir': '/home/dan/work/light-head-rcnn/models/run02/',
    'train_dataset_path': '/mnt/datasets/COCO/train_shards/',
    'val_dataset_path': '/mnt/datasets/COCO/val_shards/',
    'pretrained_checkpoint': 'pretrained/shufflenet_v2_1.0x/model.ckpt-1661328',
    'num_classes': 80,
    'positives_threshold': 0.7,
    'backbone': 'shufflenet',
    'depth_multiplier': 1.0,
    'weight_decay': 1e-5,
    'channels_middle': 64,
    'aspect_ratios': [0.5, 1.0, 2.0],
    'scales': [32, 64, 128, 256, 512],
    'rpn_num_channels': 256,
    'channels_middle': 64,
    'fc_layer_size': 2048,
    'max_boxes_per_class': 25,
    'num_steps': 900000,
    'lr_boundaries': [500000, 800000],
    'lr_values': [7e-4, 7e-5, 7e-6],
})
