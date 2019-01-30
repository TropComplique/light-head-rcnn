import tensorflow as tf
import os
from model import model_fn, RestoreMovingAverageHook
from detector.input_pipeline import Pipeline

# to train an object detector on coco use this:
# from params import coco_params as params

# to train a LIGHT object detector on coco use this:
from params import coco_light_params as params

tf.logging.set_verbosity('INFO')


"""
The purpose of this script is to train a detector.
Evaluation will happen periodically.

To use it just run:
python train.py
"""

GPU_TO_USE = '0'


def get_input_fn(is_training):

    dataset_path = params['train_dataset_path'] if is_training else params['val_dataset_path']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        pipeline = Pipeline(filenames, is_training, params)
        return pipeline.dataset

    return input_fn


session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=params['model_dir'], session_config=session_config,
    save_summary_steps=1000, save_checkpoints_secs=1800,
    log_step_count_steps=1000
)


if params['backbone'] == 'resnet':
    scope_to_restore = 'resnet_v1_50/'
elif params['backbone'] == 'mobilenet':
    scope_to_restore = 'MobilenetV1/'
elif params['backbone'] == 'shufflenet':
    scope_to_restore = 'ShuffleNetV2/'
warm_start = tf.estimator.WarmStartSettings(
    params['pretrained_checkpoint'], [scope_to_restore]
)


train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(
    model_fn, params=params,
    config=run_config,
    warm_start_from=warm_start
)


train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=params['num_steps'])
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None, start_delay_secs=3600 * 3, throttle_secs=3600 * 3,
    hooks=[RestoreMovingAverageHook(params['model_dir'])]
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
