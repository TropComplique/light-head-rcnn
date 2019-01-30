
coco_params = {

    'model_dir': 'models/run00/',
    'train_dataset_path': '~/datasets/COCO/train_shards/',
    'val_dataset_path': '~/datasets/COCO/val_shards/',
    'pretrained_checkpoint': 'pretrained/resnet_v1_50.ckpt',
    'num_classes': 80,

    # IMAGE SIZES:

    # an integer or None
    'evaluation_min_dimension': 800,
    # an integer or None
    'evaluation_max_dimension': 1200,

    # a list of integers
    'training_min_dimensions': [600, 700, 800, 900, 1000],
    # an integer or None
    'training_max_dimension': 1400,

    'num_steps': 2500000,
    'initial_learning_rate': 3e-4,

    # PROPOSAL GENERATION:

    # a float number
    'before_nms_score_threshold': 1e-6,
    # an integer
    'nms_max_output_size': 2000,
    # a float number
    'proposal_iou_threshold': 0.7,
    # an integer
    'rpn_num_channels': 512,

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

    'positives_threshold': 0.7,
    'negatives_threshold': 0.3,
    'second_stage_threshold': 0.5,

    # FINAL POSTPROCESSING:

    'score_threshold': 0.15,
    'iou_threshold': 0.6,
    'max_boxes_per_class': 20,

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
    'channels_middle': 256,
    # an integer
    'fc_layer_size': 2048,

    # ANCHOR GENERATION:

    # a list of integers
    'scales': [32, 64, 128, 256, 512],
    # a list of float numbers
    'aspect_ratios': [1.0, 2.0, 0.5],
    # a tuple of integers
    'anchor_stride': (16, 16),
    # a tuple of integers
    'anchor_offset': (0, 0),
}

coco_light_params = coco_params.copy()
coco_light_params.update({
    'model_dir': 'models/run01/',
    'train_dataset_path': '/mnt/datasets/COCO/train_shards/',
    'val_dataset_path': '/mnt/datasets/COCO/val_shards/',
    'pretrained_checkpoint': 'pretrained/shufflenet_v2_1.0x/model.ckpt-1661328',
    'backbone': 'shufflenet',
    'depth_multiplier': 1.0,
    'rpn_num_channels': 256,
    'channels_middle': 64,
    'training_min_dimensions': [700, 800, 900],
    'training_max_dimension': 1200,
})
