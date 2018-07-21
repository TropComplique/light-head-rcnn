
params_wider = {
    model_dir
    train_dataset_path
    val_dataset_path
    num_steps
    'num_classes': 1,
    'labels': '/home/dan/work/light-head-rcnn/data/wider_labels.txt',
    # head
    'p': 7, # an integer.
    'k': 15,  # an integer.
    'channels_middle': 256, # an integer.
    'crop_size': (14, 14),
    'num_spatial_bins': (7, 7),

    'scales': [16, 32, 64, 128, 256, 512],
    'aspect_ratios': [0.5, 1.0, 2.0],
    'anchor_stride': (16, 16),
    'anchor_offset': (0, 0),

    'nms_max_output_size': 300,
    'iou_threshold': 0.7,
    
    # a list of integers
    'training_min_dimensions': [600, 700, 800, 900, 1000],

    # an integer or None
    'training_max_dimension': 1400,  
    
    # an integer or None
    'evaluation_min_dimension': 800,  
     
    # an integer or None
    'evaluation_max_dimension': 1200,  
    
    score_threshold
    'iou_threshold'],
            max_boxes=params['max_boxes']
    weight_decay
    'lr_boundaries'], params['lr_values'
                             
                             {
  "model_params": {
    "model_dir": "models/run00",

    "weight_decay": 1e-3,
    "score_threshold": 0.05, "iou_threshold": 0.3, "max_boxes": 200,

    "localization_loss_weight": 1.0, "classification_loss_weight": 1.0,

    "loss_to_use": "classification",
    "loc_loss_weight": 0.0, "cls_loss_weight": 1.0,
    "num_hard_examples": 500, "nms_threshold": 0.99,
    "max_negatives_per_positive": 3.0, "min_negatives_per_image": 30,

    "lr_boundaries": [160000, 200000],
    "lr_values": [0.004, 0.0004, 0.00004]
  },

  "input_pipeline_params": {
    "image_size": [1024, 1024],
    "batch_size": 16,
    "train_dataset": "data/train_shards/",
    "val_dataset": "data/val_shards/",
    "num_steps": 240000
  }
}


}

