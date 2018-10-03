import tensorflow as tf
import os
import shutil
from model import model_fn
from detector.input_pipeline.pipeline import resize_keeping_aspect_ratio
from params import coco_light_params as params


"""
The purpose of this script is to export
the inference graph as a SavedModel.

Also it creates a .pb frozen inference graph.
"""


OUTPUT_FOLDER = 'export/'  # for savedmodel
GPU_TO_USE = '1'
PB_FILE_PATH = 'model.pb'
RESIZE = True
MIN_DIMENSION = 800
MAX_DIMENSION = 1200
WIDTH, HEIGHT = None, None
NMS_MAX_OUTPUT_SIZE = 100
BATCH_SIZE = 1  # must be an integer
assert BATCH_SIZE == 1


def export_savedmodel():
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = GPU_TO_USE
    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(
        model_dir=params['model_dir'],
        session_config=config
    )
    params['nms_max_output_size'] = NMS_MAX_OUTPUT_SIZE
    estimator = tf.estimator.Estimator(model_fn, params=params, config=run_config)

    def serving_input_receiver_fn():
        raw_images = tf.placeholder(dtype=tf.uint8, shape=[BATCH_SIZE, None, None, 3], name='images')
        w, h = tf.shape(raw_images)[2], tf.shape(raw_images)[1]

        with tf.device('/gpu:0'):

            images = tf.to_float(raw_images)
            if RESIZE:
                images = tf.squeeze(images, 0)
                images = resize_keeping_aspect_ratio(images, MIN_DIMENSION, MAX_DIMENSION)
                images = tf.expand_dims(images, 0)

            features = {
                'images': (1.0/255.0) * images,
                'images_size': tf.stack([w, h])
            }
        return tf.estimator.export.ServingInputReceiver(features, {'images': raw_images})

    shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
    os.mkdir(OUTPUT_FOLDER)
    estimator.export_savedmodel(OUTPUT_FOLDER, serving_input_receiver_fn)


def convert_to_pb():

    subfolders = os.listdir(OUTPUT_FOLDER)
    assert len(subfolders) == 1
    last_saved_model = os.path.join(OUTPUT_FOLDER, subfolders[0])

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = GPU_TO_USE

    with graph.as_default():
        with tf.Session(graph=graph, config=config) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], last_saved_model)

            # output ops
            keep_nodes = ['boxes', 'labels', 'scores', 'num_boxes_per_image']

            input_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(),
                output_node_names=keep_nodes
            )
            output_graph_def = tf.graph_util.remove_training_nodes(
                input_graph_def, protected_nodes=keep_nodes
            )

            with tf.gfile.GFile(PB_FILE_PATH, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))


tf.logging.set_verbosity('INFO')
export_savedmodel()
convert_to_pb()
