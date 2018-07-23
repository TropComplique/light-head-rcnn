import tensorflow as tf
import os
from model import model_fn
from params import wider_params as params


"""
The purpose of this script is to export
the inference graph as a SavedModel.

Also it creates a .pb frozen inference graph.
"""


OUTPUT_FOLDER = 'export/run00'
GPU_TO_USE = '1'

WIDTH, HEIGHT = None, None
BATCH_SIZE = 1
# size of an input image,
# set (None, None) if you want inference
# for images of variable size


tf.logging.set_verbosity('INFO')


config = tf.ConfigProto()
config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=params['model_dir'],
    session_config=config
)
estimator = tf.estimator.Estimator(model_fn, params=params, config=run_config)


def serving_input_receiver_fn():
    images = tf.placeholder(dtype=tf.uint8, shape=[BATCH_SIZE, None, None, 3], name='images')
    features = {'images': tf.to_float(images)}
    return tf.estimator.export.ServingInputReceiver(features, {'images': images})


estimator.export_savedmodel(
    OUTPUT_FOLDER, serving_input_receiver_fn
)


def convert_to_pb(saved_model_folder):

    subfolders = os.listdir(saved_model_folder)
    last_saved_model = os.path.join(saved_model_folder, sorted(subfolders)[0])

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
            print([n.name for n in input_graph_def.node if 'NonMaxSuppression' in n.name])
            output_graph_def = tf.graph_util.remove_training_nodes(
                input_graph_def, protected_nodes=keep_nodes
            )

            with tf.gfile.GFile('model.pb', 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))


convert_to_pb(OUTPUT_FOLDER)
