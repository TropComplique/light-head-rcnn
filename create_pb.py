import tensorflow as tf
from model import model_fn

"""
The purpose of this script is to export
the inference graph as a SavedModel.

Also it creates a .pb frozen inference graph.
"""

CONFIG = 'config.json'
OUTPUT_FOLDER = 'export/run00'
GPU_TO_USE = '0'

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
    images = tf.placeholder(dtype=tf.uint8, shape=[BATCH_SIZE, HEIGHT, WIDTH, 3], name='images')
    features = {'images': tf.to_float(images)}
    return tf.estimator.export.ServingInputReceiver(features, {'images': images})


estimator.export_savedmodel(
    OUTPUT_FOLDER, serving_input_receiver_fn
)


def convert_to_pb(saved_model_folder):

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = GPU_TO_USE

    with graph.as_default():
        with tf.Session(graph=graph, config=config) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_folder)

            # output ops
            keep_nodes = ['boxes', 'labels', 'scores', 'num_boxes']

            input_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(),
                output_node_names=keep_nodes
            )
            output_graph_def = tf.graph_util.remove_training_nodes(
                input_graph_def,
                protected_nodes=keep_nodes + [n.name for n in input_graph_def.node if 'nms' in n.name]
            )
            # ops in 'nms' scope must be protected for some reason,
            # but why?

            with tf.gfile.GFile(ARGS.output_pb, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))


ARGS = make_args()
tf.logging.set_verbosity('INFO')
main()
