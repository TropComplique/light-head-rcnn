import tensorflow as tf
import os
import shutil
from model import model_fn
from detector.input_pipeline.pipeline import resize_keeping_aspect_ratio

# from params import wider_light_params as params
from params import coco_params as params


"""
The purpose of this script is to export
the inference graph as a SavedModel.

Also it creates a .pb frozen inference graph.
"""


OUTPUT_FOLDER = 'export/'  # for savedmodel
GPU_TO_USE = '0'
PB_FILE_PATH = 'model.pb'
MIN_DIMENSION = 800
MAX_DIMENSION = 1200
WIDTH, HEIGHT = None, None
NMS_MAX_OUTPUT_SIZE = 100
BATCH_SIZE = 1  # must be an integer


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
        images = tf.placeholder(dtype=tf.uint8, shape=[BATCH_SIZE, None, None, 3], name='images')
        w, h = tf.shape(images)[2], tf.shape(images)[1]
        
        with tf.device('/gpu:0'):
    #         def fn(image):
    #             return resize_keeping_aspect_ratio(image, MIN_DIMENSION, MAX_DIMENSION)
            images = tf.to_float(images)
            resized_images = tf.expand_dims(resize_keeping_aspect_ratio(tf.squeeze(images, 0), MIN_DIMENSION, MAX_DIMENSION), 0)
            #resized_images = tf.map_fn(fn, tf.to_float(images), dtype=tf.float32, back_prop=False)

            features = {
                'images': (1.0/255.0) * resized_images,
                'images_size': tf.stack([w, h])
            }
        return tf.estimator.export.ServingInputReceiver(features, {'images': images})

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
