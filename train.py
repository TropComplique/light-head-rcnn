import tensorflow as tf
import os
from model import model_fn
from detector.input_pipeline import Pipeline

# to train a face detector use this:
#from params import wider_params as params

# to train an object detector on coco use this:
# from params import coco_params as params

from params import wider_light_params as params

tf.logging.set_verbosity('INFO')


"""
The purpose of this script is to train a detector.
Evaluation will happen periodically.

To use it just run:
python train.py
"""

GPU_TO_USE = '1'


def get_input_fn(is_training):

    dataset_path = params['train_dataset_path'] if is_training else params['val_dataset_path']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(filenames, is_training, params)
            return pipeline.dataset

    return input_fn


session_config = tf.ConfigProto()
session_config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=params['model_dir'], session_config=session_config,
    save_summary_steps=200, save_checkpoints_secs=600,
    log_step_count_steps=100
)

train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, params=params, config=run_config)

train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=params['num_steps'])
#eval_spec = tf.estimator.EvalSpec(val_input_fn, steps=None, start_delay_secs=10, throttle_secs=10)
eval_spec = tf.estimator.EvalSpec(val_input_fn, steps=None, start_delay_secs=2*3600, throttle_secs=2*3600)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
